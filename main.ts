import { serve } from "https://deno.land/std@0.208.0/http/server.ts";

// Types
interface ChatMessage {
  role: string;
  content: string;
}

interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  stream?: boolean;
  temperature?: number;
}

interface ModelInfo {
  id: string;
  object: string;
  created: number;
  owned_by: string;
}

interface ModelList {
  object: string;
  data: ModelInfo[];
}

interface ChatCompletionChoice {
  message: ChatMessage;
  index: number;
  finish_reason: string;
}

interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface StreamChoice {
  delta: Record<string, any>;
  index: number;
  finish_reason?: string;
}

interface StreamResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: StreamChoice[];
}

// JWT Token management
class TokenManager {
  private token: string | null = null;
  private tokenExpiry: number = 0;
  private tokenRefreshPromise: Promise<string> | null = null;

  async getValidToken(): Promise<string> {
    const now = Date.now();
    
    // If token exists and hasn't expired, return it
    if (this.token && now < this.tokenExpiry) {
      return this.token;
    }

    // If refresh is already in progress, wait for it
    if (this.tokenRefreshPromise) {
      return await this.tokenRefreshPromise;
    }

    // Start token refresh
    this.tokenRefreshPromise = this.refreshToken();
    
    try {
      const newToken = await this.tokenRefreshPromise;
      return newToken;
    } finally {
      this.tokenRefreshPromise = null;
    }
  }

  private async refreshToken(): Promise<string> {
    console.log("Refreshing JWT token...");
    
    try {
      // First, visit the guest page to get session
      const guestResponse = await fetch("https://www.k2think.ai/guest", {
        method: "GET",
        headers: {
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
          "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
          "Accept-Language": "en-US,en;q=0.5",
          "Accept-Encoding": "gzip, deflate, br",
          "DNT": "1",
          "Connection": "keep-alive",
          "Upgrade-Insecure-Requests": "1",
        },
      });

      const cookies = guestResponse.headers.get("set-cookie");
      
      // Try to get JWT token from API endpoint
      const tokenResponse = await fetch("https://www.k2think.ai/api/guest/token", {
        method: "POST",
        headers: {
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
          "Content-Type": "application/json",
          "Accept": "application/json",
          "Origin": "https://www.k2think.ai",
          "Referer": "https://www.k2think.ai/guest",
          "Cookie": cookies || "",
        },
        body: JSON.stringify({}),
      });

      if (tokenResponse.ok) {
        const tokenData = await tokenResponse.json();
        if (tokenData.token) {
          this.token = tokenData.token;
          // Set expiry to 55 minutes from now (tokens usually expire in 1 hour)
          this.tokenExpiry = Date.now() + 55 * 60 * 1000;
          console.log("JWT token refreshed successfully");
          return this.token;
        }
      }

      // Fallback: try to extract token from page HTML
      if (guestResponse.ok) {
        const html = await guestResponse.text();
        const tokenMatch = html.match(/(?:token|jwt)["']\s*:\s*["']([^"']+)["']/i);
        if (tokenMatch) {
          this.token = tokenMatch[1];
          this.tokenExpiry = Date.now() + 55 * 60 * 1000;
          console.log("JWT token extracted from HTML");
          return this.token;
        }
      }

      throw new Error("Failed to obtain JWT token");
    } catch (error) {
      console.error("Error refreshing token:", error);
      throw error;
    }
  }
}

// Global token manager instance
const tokenManager = new TokenManager();

// Helper functions
function generateId(): string {
  return `chatcmpl-${crypto.randomUUID().replace(/-/g, "")}`;
}

function getCurrentTimestamp(): number {
  return Math.floor(Date.now() / 1000);
}

function extractReasoningAndAnswer(content: string): [string, string] {
  if (!content) return ["", ""];
  
  let reasoning = "";
  let answer = "";
  
  // Extract reasoning part - inside <details> tags
  const reasoningPattern = /<details type="reasoning"[^>]*>.*?<summary>.*?<\/summary>(.*?)<\/details>/s;
  const reasoningMatch = content.match(reasoningPattern);
  if (reasoningMatch) {
    reasoning = reasoningMatch[1].trim();
  }
  
  // Extract answer part - inside <answer> tags
  const answerPattern = /<answer>(.*?)<\/answer>/s;
  const answerMatch = content.match(answerPattern);
  if (answerMatch) {
    answer = answerMatch[1].trim();
  }
  
  return [reasoning, answer];
}

function calculateDeltaContent(previousContent: string, currentContent: string): string {
  return currentContent.slice(previousContent.length);
}

function extractContentFromJson(obj: any): [string, boolean, any, string | null] {
  if (!obj || typeof obj !== "object") {
    return ["", false, null, null];
  }
  
  // Usage payload
  if (obj.usage) {
    return ["", false, obj.usage, null];
  }
  
  // Done marker
  if (obj.done === true) {
    return ["", true, obj.usage || null, null];
  }
  
  // OpenAI-like chunk
  if (Array.isArray(obj.choices) && obj.choices.length > 0) {
    const delta = obj.choices[0].delta || {};
    const role = delta.role;
    const contentPiece = delta.content || "";
    return [contentPiece, false, null, role];
  }
  
  // Raw content packet
  if (typeof obj.content === "string") {
    return [obj.content, false, null, null];
  }
  
  return ["", false, null, null];
}

async function* streamGeneratorK2(
  payload: any,
  headers: Record<string, string>,
  model: string
): AsyncGenerator<string> {
  const streamId = generateId();
  const createdTime = getCurrentTimestamp();
  let accumulatedContent = "";
  let previousReasoning = "";
  let previousAnswer = "";
  let reasoningPhase = true;

  // Emit initial role
  const initialResponse: StreamResponse = {
    id: streamId,
    object: "chat.completion.chunk",
    created: createdTime,
    model: model,
    choices: [{ delta: { role: "assistant" }, index: 0 }],
  };
  yield `data: ${JSON.stringify(initialResponse)}\n\n`;

  try {
    const response = await fetch("https://www.k2think.ai/api/guest/chat/completions", {
      method: "POST",
      headers: headers,
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("Failed to get response reader");
    }

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        if (!line.startsWith("data:")) continue;
        
        const dataStr = line.slice(5).trim();
        if (!dataStr || dataStr === "-1") continue;
        if (["[DONE]", "DONE", "done"].includes(dataStr)) {
          return;
        }

        let contentPiece = "";
        
        try {
          const obj = JSON.parse(dataStr);
          const [piece, isDone] = extractContentFromJson(obj);
          if (isDone) return;
          contentPiece = piece;
        } catch {
          contentPiece = dataStr;
        }

        if (contentPiece) {
          accumulatedContent = contentPiece;
          
          const [currentReasoning, currentAnswer] = extractReasoningAndAnswer(accumulatedContent);
          
          // Send reasoning delta if in reasoning phase
          if (reasoningPhase && currentReasoning) {
            const reasoningDelta = calculateDeltaContent(previousReasoning, currentReasoning);
            if (reasoningDelta.trim()) {
              const reasoningResponse: StreamResponse = {
                id: streamId,
                object: "chat.completion.chunk",
                created: createdTime,
                model: model,
                choices: [{ delta: { reasoning_content: reasoningDelta }, index: 0 }],
              };
              yield `data: ${JSON.stringify(reasoningResponse)}\n\n`;
              previousReasoning = currentReasoning;
            }
          }
          
          // Check if we've moved to answer phase
          if (currentAnswer && reasoningPhase) {
            reasoningPhase = false;
            // Send any remaining reasoning content
            if (currentReasoning && currentReasoning !== previousReasoning) {
              const reasoningDelta = calculateDeltaContent(previousReasoning, currentReasoning);
              if (reasoningDelta.trim()) {
                const reasoningResponse: StreamResponse = {
                  id: streamId,
                  object: "chat.completion.chunk",
                  created: createdTime,
                  model: model,
                  choices: [{ delta: { reasoning_content: reasoningDelta }, index: 0 }],
                };
                yield `data: ${JSON.stringify(reasoningResponse)}\n\n`;
              }
            }
          }
          
          // Send answer delta if in answer phase
          if (!reasoningPhase && currentAnswer) {
            const answerDelta = calculateDeltaContent(previousAnswer, currentAnswer);
            if (answerDelta.trim()) {
              const answerResponse: StreamResponse = {
                id: streamId,
                object: "chat.completion.chunk",
                created: createdTime,
                model: model,
                choices: [{ delta: { content: answerDelta }, index: 0 }],
              };
              yield `data: ${JSON.stringify(answerResponse)}\n\n`;
              previousAnswer = currentAnswer;
            }
          }
        }
      }
    }
  } catch (error) {
    console.error("Stream error:", error);
  }

  // Close stream
  const finalResponse: StreamResponse = {
    id: streamId,
    object: "chat.completion.chunk",
    created: createdTime,
    model: model,
    choices: [{ delta: {}, index: 0, finish_reason: "stop" }],
  };
  yield `data: ${JSON.stringify(finalResponse)}\n\n`;
  yield "data: [DONE]\n\n";
}

// Environment variables
const VALID_CLIENT_KEYS = new Set(
  Deno.env.get("CLIENT_API_KEYS")?.split(",") || ["sk-talkai-default"]
);

// Authentication middleware
function authenticateClient(request: Request): boolean {
  if (VALID_CLIENT_KEYS.size === 0) return true;
  
  const authHeader = request.headers.get("Authorization");
  if (!authHeader) return false;
  
  const token = authHeader.replace("Bearer ", "");
  return VALID_CLIENT_KEYS.has(token);
}

// Models endpoint
function getModelsList(): ModelList {
  const defaultModels = ["MBZUAI-IFM/K2-Think"];
  const models = Deno.env.get("MODELS")?.split(",") || defaultModels;
  
  return {
    object: "list",
    data: models.map(modelId => ({
      id: modelId,
      object: "model",
      created: getCurrentTimestamp(),
      owned_by: "talkai"
    }))
  };
}

// Main request handler
async function handler(request: Request): Promise<Response> {
  const url = new URL(request.url);
  
  // CORS headers
  const corsHeaders = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
  };
  
  if (request.method === "OPTIONS") {
    return new Response(null, { status: 200, headers: corsHeaders });
  }
  
  // Health check
  if (url.pathname === "/" || url.pathname === "/health") {
    return new Response(JSON.stringify({ status: "ok", service: "K2Think API Adapter" }), {
      status: 200,
      headers: { ...corsHeaders, "Content-Type": "application/json" }
    });
  }
  
  // Models endpoint
  if (url.pathname === "/v1/models" && request.method === "GET") {
    if (!authenticateClient(request)) {
      return new Response(JSON.stringify({ error: "Invalid API key" }), {
        status: 401,
        headers: { ...corsHeaders, "Content-Type": "application/json" }
      });
    }
    
    return new Response(JSON.stringify(getModelsList()), {
      status: 200,
      headers: { ...corsHeaders, "Content-Type": "application/json" }
    });
  }
  
  // Chat completions endpoint
  if (url.pathname === "/v1/chat/completions" && request.method === "POST") {
    if (!authenticateClient(request)) {
      return new Response(JSON.stringify({ error: "Invalid API key" }), {
        status: 401,
        headers: { ...corsHeaders, "Content-Type": "application/json" }
      });
    }
    
    try {
      const chatRequest: ChatCompletionRequest = await request.json();
      
      if (!chatRequest.messages || chatRequest.messages.length === 0) {
        return new Response(JSON.stringify({ error: "Messages required" }), {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" }
        });
      }
      
      // Get JWT token
      let jwtToken: string;
      try {
        jwtToken = await tokenManager.getValidToken();
      } catch (error) {
        console.error("Failed to get JWT token:", error);
        return new Response(JSON.stringify({ error: "Failed to authenticate with K2Think API" }), {
          status: 500,
          headers: { ...corsHeaders, "Content-Type": "application/json" }
        });
      }
      
      // Build K2Think-compatible message list
      const k2Messages: { role: string; content: string }[] = [];
      let systemPrompt = "";
      
      for (const msg of chatRequest.messages) {
        if (msg.role === "system") {
          systemPrompt = msg.content;
        } else if (msg.role === "user" || msg.role === "assistant") {
          k2Messages.push({ role: msg.role, content: msg.content });
        }
      }
      
      // Merge system prompt into first user message
      if (systemPrompt) {
        const firstUserIndex = k2Messages.findIndex(m => m.role === "user");
        if (firstUserIndex !== -1) {
          k2Messages[firstUserIndex].content = `${systemPrompt}\n\n${k2Messages[firstUserIndex].content}`;
        } else {
          k2Messages.unshift({ role: "user", content: systemPrompt });
        }
      }
      
      const payload = {
        stream: true,
        model: chatRequest.model,
        messages: k2Messages,
        params: {}
      };
      
      const headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Authorization": `Bearer ${jwtToken}`,
        "Origin": "https://www.k2think.ai",
        "Referer": "https://www.k2think.ai/guest",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
      };
      
      // Always return streaming response
      const stream = streamGeneratorK2(payload, headers, chatRequest.model);
      
      return new Response(
        new ReadableStream({
          async start(controller) {
            const encoder = new TextEncoder();
            try {
              for await (const chunk of stream) {
                controller.enqueue(encoder.encode(chunk));
              }
            } catch (error) {
              console.error("Stream error:", error);
              controller.enqueue(encoder.encode(`data: {"error": "Stream error"}\n\n`));
            } finally {
              controller.close();
            }
          }
        }),
        {
          status: 200,
          headers: {
            ...corsHeaders,
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
          }
        }
      );
      
    } catch (error) {
      console.error("Request error:", error);
      return new Response(JSON.stringify({ error: "Internal server error" }), {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" }
      });
    }
  }
  
  return new Response(JSON.stringify({ error: "Not found" }), {
    status: 404,
    headers: { ...corsHeaders, "Content-Type": "application/json" }
  });
}

// Start server
console.log("🚀 K2Think API Adapter starting...");
serve(handler, { port: 8000 });
