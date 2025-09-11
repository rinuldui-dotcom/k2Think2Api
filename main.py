import json
import os
import time
import uuid
import re
from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple
from html import unescape

import httpx
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = 0.7


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "talkai"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class ChatCompletionChoice(BaseModel):
    message: ChatMessage
    index: int = 0
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


class StreamChoice(BaseModel):
    delta: Dict[str, Any] = Field(default_factory=dict)
    index: int = 0
    finish_reason: Optional[str] = None


class StreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]


app = FastAPI(title="TalkAI OpenAI API Adapter")
security = HTTPBearer()
VALID_CLIENT_KEYS: set = set()


def load_client_api_keys():
    global VALID_CLIENT_KEYS
    try:
        with open("client_api_keys.json", "r", encoding="utf-8") as f:
            keys = json.load(f)
            VALID_CLIENT_KEYS = set(keys) if isinstance(keys, list) else set()
    except:
        VALID_CLIENT_KEYS = set()


async def authenticate_client(auth: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if VALID_CLIENT_KEYS and (not auth or auth.credentials not in VALID_CLIENT_KEYS):
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.on_event("startup")
async def startup():
    load_client_api_keys()


def get_models_list() -> ModelList:
    try:
        with open("models.json", "r", encoding="utf-8") as f:
            models_dict = json.load(f)
        return ModelList(data=[ModelInfo(id=model_id) for model_id in models_dict.values()])
    except:
        return ModelList(data=[ModelInfo(id="MBZUAI-IFM/K2-Think")])


@app.get("/v1/models", response_model=ModelList)
async def list_models(auth: Optional[HTTPAuthorizationCredentials] = Depends(authenticate_client)):
    return get_models_list()


def extract_reasoning_and_answer(content: str) -> Tuple[str, str]:
    """Extract reasoning content and answer content from K2Think response.
    Returns (reasoning_content, answer_content)
    """
    if not content:
        return "", ""
    
    reasoning = ""
    answer = ""
    
    # 提取reasoning部分 - 在<details>标签内
    reasoning_pattern = r'<details type="reasoning"[^>]*>.*?<summary>.*?</summary>(.*?)</details>'
    reasoning_match = re.search(reasoning_pattern, content, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # 提取answer部分 - 在<answer>标签内
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, content, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    
    return reasoning, answer

def calculate_delta_content(previous_content: str, current_content: str) -> str:
    """Calculate the delta (new content) between previous and current content."""
    return current_content[len(previous_content):]


def _extract_content_from_json(obj: Dict[str, Any]) -> Tuple[str, bool, Optional[Dict[str, Any]], Optional[str]]:
    """Extract content piece and meta from a K2Think SSE JSON object.
    Returns (content_piece, is_done, usage, role)
    """
    if not isinstance(obj, dict):
        return "", False, None, None
    # Usage payload
    if obj.get("usage"):
        return "", False, obj.get("usage"), None
    # Done marker (may also include final content)
    if obj.get("done") is True:
        # Some responses include full content in done packet; we won't forward it to avoid duplication
        return "", True, obj.get("usage"), None
    # OpenAI-like chunk
    if isinstance(obj.get("choices"), list) and obj["choices"]:
        delta = obj["choices"][0].get("delta") or {}
        role = delta.get("role")
        content_piece = delta.get("content") or ""
        return content_piece, False, None, role
    # Raw content packet
    if isinstance(obj.get("content"), str):
        return obj["content"], False, None, None
    return "", False, None, None


async def stream_generator_k2(payload: Dict[str, Any], headers: Dict[str, str], model: str) -> AsyncGenerator[str, None]:
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())
    role_sent = False
    accumulated_content = ""
    previous_reasoning = ""
    previous_answer = ""
    reasoning_phase = True

    # Emit initial role
    yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={'role': 'assistant'})]).json()}\n\n"
    role_sent = True

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST",
            "https://www.k2think.ai/api/guest/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                # K2Think uses SSE lines prefixed with 'data:'
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "-1":
                    continue
                if data_str in ("[DONE]", "DONE", "done"):
                    break

                content_piece: str = ""
                role: Optional[str] = None

                # Try to parse as JSON; fallback to raw string
                try:
                    obj = json.loads(data_str)
                    content_piece, is_done, _usage, role = _extract_content_from_json(obj)
                    if is_done:
                        break
                except Exception:
                    content_piece = data_str

                # Accumulate content and process
                if content_piece:
                    accumulated_content = content_piece
                    
                    # Extract reasoning and answer from accumulated content
                    current_reasoning, current_answer = extract_reasoning_and_answer(accumulated_content)
                
                    # Send reasoning delta if in reasoning phase
                    if reasoning_phase and current_reasoning:
                        reasoning_delta = calculate_delta_content(previous_reasoning, current_reasoning)
                        if reasoning_delta.strip():
                            yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={'reasoning_content': reasoning_delta})]).json()}\n\n"
                            previous_reasoning = current_reasoning
                    
                    # Check if we've moved to answer phase
                    if current_answer and reasoning_phase:
                        reasoning_phase = False
                        # Send any remaining reasoning content
                        if current_reasoning and current_reasoning != previous_reasoning:
                            reasoning_delta = calculate_delta_content(previous_reasoning, current_reasoning)
                            if reasoning_delta.strip():
                                yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={'reasoning_content': reasoning_delta})]).json()}\n\n"
                    
                    # Send answer delta if in answer phase
                    if not reasoning_phase and current_answer:
                        answer_delta = calculate_delta_content(previous_answer, current_answer)
                        if answer_delta.strip():
                            yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={'content': answer_delta})]).json()}\n\n"
                            previous_answer = current_answer

    # Close stream
    yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={}, finish_reason='stop')]).json()}\n\n"
    yield "data: [DONE]\n\n"


async def aggregate_stream(response: httpx.Response) -> str:
    pieces: List[str] = []
    async for line in response.aiter_lines():
        if not line or not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if not data_str or data_str in ("-1", "[DONE]", "DONE", "done"):
            continue
        try:
            obj = json.loads(data_str)
            content_piece, is_done, _usage, _role = _extract_content_from_json(obj)
            if is_done:
                continue
        except Exception:
            content_piece = data_str
        if content_piece:
            pieces.append(content_piece)
    
    # Extract only the answer content for non-streaming responses
    accumulated_content = "".join(pieces)
    _reasoning, answer_content = extract_reasoning_and_answer(accumulated_content)
    return answer_content.replace("\\n", "\n")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, auth: Optional[HTTPAuthorizationCredentials] = Depends(authenticate_client)):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages required")
    
    # Build K2Think-compatible message list and merge system prompt into first user message
    k2_messages: List[Dict[str, str]] = []
    system_prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role in ("user", "assistant"):
            k2_messages.append({"role": msg.role, "content": msg.content})
    if system_prompt:
        # Find first user message to prepend system prompt
        for m in k2_messages:
            if m.get("role") == "user":
                m["content"] = f"{system_prompt}\n\n{m['content']}"
                break
        else:
            # No user message exists, create one
            k2_messages.insert(0, {"role": "user", "content": system_prompt})

    # K2Think guest chat completions endpoint (SSE streaming)
    payload = {
        "stream": True,  # always request stream; we will aggregate if client asks non-stream
        "model": request.model,
        "messages": k2_messages,
        "params": {}
    }

    headers = {
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
    }

    try:
       
        return StreamingResponse(stream_generator_k2(payload, headers, request.model), media_type="text/event-stream")
        

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail="K2Think API error")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal error")


if __name__ == "__main__":
    import uvicorn
    
    if not os.path.exists("client_api_keys.json"):
        with open("client_api_keys.json", "w", encoding="utf-8") as f:
            json.dump([f"sk-talkai-{uuid.uuid4().hex}"], f)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
