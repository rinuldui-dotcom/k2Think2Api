// TalkAI OpenAI API Adapter for Deno Deploy
import { serve } from "https://deno.land/std@0.208.0/http/server.ts";

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
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}

interface StreamChoice {
  delta: Record<string, any>;
  index: number;
  finish_reason: string | null;
}

interface StreamResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: StreamChoice[];
}

// Configuration - Use environment variables in Deno Deploy
const VALID_CLIENT_KEYS = new Set(
  Deno.env.get("CLIENT_API_KEYS")?.split(",") || ["sk-talkai-default"]
);

const DEFAULT_MODELS = ["MBZUAI-IFM/K2-Think"];

function generateId(): string {
  return `chatcmpl-${crypto.randomUUID().replace(/-/g, "")}`;
}

function authenticateClient(request: Request): boolean {
  if (VALID_CLIENT_KEYS.size === 0) return true;
  
  const auth = request.headers.get("authorization");
  if (!auth || !auth.startsWith("Bearer ")) return false;
  
  const token = auth.slice(7);
  return VALID_CLIENT_KEYS.has(token);
}

function getModelsList(): ModelList {
  const modelsStr = Deno.env.get("MODELS_JSON");
  let models = DEFAULT_MODELS;
  
  if (modelsStr) {
    try {
      const modelsDict = JSON.parse(modelsStr);
      models = Object.values(modelsDict) as string[];
    } catch {
      models = DEFAULT_MODELS;
    }
  }
  
  return {
    object: "list",
    data: models.map(id => ({
      id,
      object: "model",
      created: Math.floor(Date.now() / 1000),
      owned_by: "talkai"
    }))
  };
}

function extractReasoningAndAnswer(content: string): [string, string] {
  if (!content) return ["", ""];
  
  let reasoning = "";
  let answer = "";
  
  // Extract reasoning content within <details> tags
  const reasoningPattern = /<details type="reasoning"[^>]*>.*?<summary>.*?<\/summary>(.*?)<\/details>/s;
  const reasoningMatch = content.match(reasoningPattern);
  if (reasoningMatch) {
    reasoning = reasoningMatch[1].trim();
  }
  
  // Extract answer content within <answer> tags
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
  if (!obj || typeof obj !== "object") return ["", false, null, null];
  
  // Usage payload
  if (obj.usage) return ["", false, obj.usage, null];
  
  // Done marker
  if (obj.done === true) return ["", true, obj.usage || null, null];
  
  // OpenAI-like chunk
  if (Array.isArray(obj.choices) && obj.choices.length > 0) {
    const delta = obj.choices[0].delta || {};
    const role = delta.role;
    const contentPiece = delta.content || "";
    return [contentPiece, false, null, role];
  }
  
  // Raw content packet
  if (typeof obj.content === "string") return [obj.content, false, null, null];
  
  return ["", false, null, null];
}

async function* streamGeneratorK2(
  payload: any, 
  headers: Record<string, string>, 
  model: string
): AsyncGenerator<string> {
  const streamId = generateId();
  const createdTime = Math.floor(Date.now() / 1000);
  let accumulatedContent = "";
  let previousReasoning = "";
  let previousAnswer = "";
  let reasoningPhase = true;

  // Emit initial role
  yield `data: ${JSON.stringify({
    id: streamId,
    object: "chat.completion.chunk",
    created: createdTime,
    model,
    choices: [{ delta: { role: "assistant" }, index: 0, finish_reason: null }]
  })}\n\n`;

  try {
    const response = await fetch("https://www.k2think.ai/api/guest/chat/completions", {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`K2Think API error: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error("No response body");

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
        if (["[DONE]", "DONE", "done"].includes(dataStr)) return;

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
              yield `data: ${JSON.stringify({
                id: streamId,
                object: "chat.completion.chunk",
                created: createdTime,
                model,
                choices: [{ delta: { reasoning_content: reasoningDelta }, index: 0, finish_reason: null }]
              })}\n\n`;
              previousReasoning = currentReasoning;
            }
          }
          
          // Check if moved to answer phase
          if (currentAnswer && reasoningPhase) {
            reasoningPhase = false;
            // Send any remaining reasoning content
            if (currentReasoning && currentReasoning !== previousReasoning) {
              const reasoningDelta = calculateDeltaContent(previousReasoning, currentReasoning);
              if (reasoningDelta.trim()) {
                yield `data: ${JSON.stringify({
                  id: streamId,
                  object: "chat.completion.chunk",
                  created: createdTime,
                  model,
                  choices: [{ delta: { reasoning_content: reasoningDelta }, index: 0, finish_reason: null }]
                })}\n\n`;
              }
            }
          }
          
          // Send answer delta if in answer phase
          if (!reasoningPhase && currentAnswer) {
            const answerDelta = calculateDeltaContent(previousAnswer, currentAnswer);
            if (answerDelta.trim()) {
              yield `data: ${JSON.stringify({
                id: streamId,
                object: "chat.completion.chunk",
                created: createdTime,
                model,
                choices: [{ delta: { content: answerDelta }, index: 0, finish_reason: null }]
              })}\n\n`;
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
  yield `data: ${JSON.stringify({
    id: streamId,
    object: "chat.completion.chunk",
    created: createdTime,
    model,
    choices: [{ delta: {}, index: 0, finish_reason: "stop" }]
  })}\n\n`;
  yield "data: [DONE]\n\n";
}

async function handleRequest(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const path = url.pathname;

  // CORS headers
  const corsHeaders = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
  };

  if (request.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  // Authentication
  if (!authenticateClient(request)) {
    return new Response(
      JSON.stringify({ error: { message: "Invalid API key", type: "invalid_request_error" } }),
      { status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }

  // Routes
  if (path === "/v1/models" && request.method === "GET") {
    return new Response(
      JSON.stringify(getModelsList()),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }

  if (path === "/v1/chat/completions" && request.method === "POST") {
    try {
      const requestData: ChatCompletionRequest = await request.json();
      
      if (!requestData.messages || requestData.messages.length === 0) {
        return new Response(
          JSON.stringify({ error: { message: "Messages required", type: "invalid_request_error" } }),
          { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      // Build K2Think-compatible message list
      const k2Messages: { role: string; content: string }[] = [];
      let systemPrompt = "";
      
      for (const msg of requestData.messages) {
        if (msg.role === "system") {
          systemPrompt = msg.content;
        } else if (msg.role === "user" || msg.role === "assistant") {
          k2Messages.push({ role: msg.role, content: msg.content });
        }
      }
      
      // Merge system prompt into first user message
      if (systemPrompt) {
        const firstUserMsg = k2Messages.find(m => m.role === "user");
        if (firstUserMsg) {
          firstUserMsg.content = `${systemPrompt}\n\n${firstUserMsg.content}`;
        } else {
          k2Messages.unshift({ role: "user", content: systemPrompt });
        }
      }

      const payload = {
        stream: true,
        model: requestData.model,
        messages: k2Messages,
        params: {}
      };

      const headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "pragma": "no-cache",
        "origin": "https://www.k2think.ai",
        "sec-fetch-site": "same-origin",
        "sec-fetch-mode": "cors",
        "sec-fetch-dest": "empty",
        "referer": "https://www.k2think.ai/guest",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,zh-TW;q=0.5",
      };

      // Always return streaming response
      const stream = new ReadableStream({
        async start(controller) {
          const encoder = new TextEncoder();
          try {
            for await (const chunk of streamGeneratorK2(payload, headers, requestData.model)) {
              controller.enqueue(encoder.encode(chunk));
            }
          } catch (error) {
            console.error("Stream error:", error);
            controller.enqueue(encoder.encode(`data: {"error": "Stream error"}\n\n`));
          } finally {
            controller.close();
          }
        }
      });

      return new Response(stream, {
        headers: {
          ...corsHeaders,
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive",
        }
      });

    } catch (error) {
      console.error("Request error:", error);
      return new Response(
        JSON.stringify({ error: { message: "Internal error", type: "internal_error" } }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }
  }

  return new Response(
    JSON.stringify({ error: { message: "Not found", type: "not_found_error" } }),
    { status: 404, headers: { ...corsHeaders, "Content-Type": "application/json" } }
  );
}

// Start the server
serve(handleRequest, { port: 8001 });

console.log("TalkAI OpenAI API Adapter running on http://localhost:8001");
console.log("Environment variables:");
console.log("- CLIENT_API_KEYS: comma-separated list of valid API keys");
console.log("- MODELS_JSON: JSON string mapping model names");
