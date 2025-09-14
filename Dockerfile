import json
import os
import time
import uuid
import re
import base64
import mimetypes
from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple, Union
from html import unescape
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import io


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = 0.7
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    max_tokens: Optional[int] = None


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


class FileUploadResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"file-{uuid.uuid4().hex}")
    object: str = "file"
    filename: str
    bytes: int
    created_at: int = Field(default_factory=lambda: int(time.time()))
    purpose: str = "assistants"


app = FastAPI(title="TalkAI OpenAI API Adapter with Function Calling")

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()
VALID_CLIENT_KEYS: set = set()

# 文件存储目录
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def load_client_api_keys():
    global VALID_CLIENT_KEYS
    try:
        with open("client_api_keys.json", "r", encoding="utf-8") as f:
            keys = json.load(f)
            VALID_CLIENT_KEYS = set(keys) if isinstance(keys, list) else set()
    except:
        VALID_CLIENT_KEYS = set()


def load_function_definitions():
    """加载函数定义配置"""
    try:
        with open("functions.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {
            "weather": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "温度单位"
                        }
                    },
                    "required": ["city"]
                }
            }
        }


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


def process_image(file_path: str) -> str:
    """处理图像文件，返回base64编码"""
    try:
        with Image.open(file_path) as img:
            # 压缩大图片
            if img.width > 1024 or img.height > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            # 转换为RGB（如果是RGBA）
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # 保存为base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")


def process_text_file(file_path: str) -> str:
    """处理文本文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 限制文件大小
            if len(content) > 50000:  # 50KB限制
                content = content[:50000] + "...[文件内容过长，已截断]"
            return content
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read()
                if len(content) > 50000:
                    content = content[:50000] + "...[文件内容过长，已截断]"
                return content
        except:
            raise HTTPException(status_code=400, detail="Unable to decode text file")


async def execute_function_call(function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """执行函数调用"""
    functions = load_function_definitions()
    
    if function_name == "get_weather":
        # 模拟天气API调用
        city = arguments.get("city", "北京")
        unit = arguments.get("unit", "celsius")
        
        # 这里可以集成真实的天气API
        mock_weather = {
            "city": city,
            "temperature": 22 if unit == "celsius" else 72,
            "unit": unit,
            "condition": "晴朗",
            "humidity": "65%",
            "wind_speed": "微风"
        }
        
        return {
            "success": True,
            "data": mock_weather
        }
    
    # 可以添加更多函数实现
    elif function_name == "web_search":
        query = arguments.get("query", "")
        # 集成搜索API的示例
        return {
            "success": True,
            "data": {
                "query": query,
                "results": [
                    {"title": f"搜索结果 for {query}", "url": "https://example.com", "snippet": "示例搜索结果"}
                ]
            }
        }
    
    else:
        return {
            "success": False,
            "error": f"Unknown function: {function_name}"
        }


@app.get("/v1/models", response_model=ModelList)
async def list_models(auth: Optional[HTTPAuthorizationCredentials] = Depends(authenticate_client)):
    return get_models_list()


@app.post("/v1/files", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    purpose: str = Form("assistants"),
    auth: Optional[HTTPAuthorizationCredentials] = Depends(authenticate_client)
):
    """文件上传接口"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # 检查文件大小 (最大10MB)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    
    # 生成唯一文件ID和保存路径
    file_id = f"file-{uuid.uuid4().hex}"
    file_extension = Path(file.filename).suffix
    save_path = UPLOAD_DIR / f"{file_id}{file_extension}"
    
    # 保存文件
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    
    return FileUploadResponse(
        id=file_id,
        filename=file.filename,
        bytes=len(content)
    )


@app.get("/v1/files/{file_id}")
async def get_file_info(
    file_id: str,
    auth: Optional[HTTPAuthorizationCredentials] = Depends(authenticate_client)
):
    """获取文件信息"""
    # 查找文件
    for file_path in UPLOAD_DIR.glob(f"{file_id}.*"):
        if file_path.exists():
            stat = file_path.stat()
            return {
                "id": file_id,
                "object": "file",
                "filename": file_path.name,
                "bytes": stat.st_size,
                "created_at": int(stat.st_ctime)
            }
    
    raise HTTPException(status_code=404, detail="File not found")


def convert_content_with_files(content: Union[str, List[Dict[str, Any]]]) -> str:
    """转换包含文件的内容为文本"""
    if isinstance(content, str):
        return content
    
    text_parts = []
    for item in content:
        if item.get("type") == "text":
            text_parts.append(item.get("text", ""))
        elif item.get("type") == "image_url":
            # 处理图片URL
            image_url = item.get("image_url", {}).get("url", "")
            if image_url.startswith("data:image"):
                text_parts.append("[图片内容已上传]")
            elif image_url.startswith("file-"):
                # 处理上传的文件
                file_id = image_url.replace("file-", "")
                for file_path in UPLOAD_DIR.glob(f"{file_id}.*"):
                    if file_path.exists():
                        mime_type, _ = mimetypes.guess_type(str(file_path))
                        if mime_type and mime_type.startswith('image/'):
                            # 处理图像文件
                            try:
                                image_b64 = process_image(str(file_path))
                                text_parts.append(f"[图片文件: {file_path.name}]")
                                # 这里可以将图片发送给支持视觉的模型
                            except:
                                text_parts.append(f"[无法处理的图片文件: {file_path.name}]")
                        else:
                            # 处理文本文件
                            try:
                                file_content = process_text_file(str(file_path))
                                text_parts.append(f"[文件 {file_path.name} 内容]:\n{file_content}")
                            except:
                                text_parts.append(f"[无法读取文件: {file_path.name}]")
                        break
                else:
                    text_parts.append(f"[文件未找到: {image_url}]")
            else:
                text_parts.append(f"[外部图片: {image_url}]")
        elif item.get("type") == "file":
            # 处理文件引用
            file_id = item.get("file_id", "")
            if file_id:
                # 与上面类似的文件处理逻辑
                for file_path in UPLOAD_DIR.glob(f"{file_id.replace('file-', '')}.*"):
                    if file_path.exists():
                        try:
                            file_content = process_text_file(str(file_path))
                            text_parts.append(f"[文件 {file_path.name} 内容]:\n{file_content}")
                        except:
                            text_parts.append(f"[无法读取文件: {file_path.name}]")
                        break
    
    return "\n".join(text_parts)


def extract_reasoning_and_answer(content: str) -> Tuple[str, str]:
    """Extract reasoning content and answer content from K2Think response."""
    if not content:
        return "", ""
    
    reasoning = ""
    answer = ""
    
    reasoning_pattern = r'<details type="reasoning"[^>]*>.*?<summary>.*?</summary>(.*?)</details>'
    reasoning_match = re.search(reasoning_pattern, content, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, content, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    
    return reasoning, answer


def calculate_delta_content(previous_content: str, current_content: str) -> str:
    """Calculate the delta (new content) between previous and current content."""
    return current_content[len(previous_content):]


def _extract_content_from_json(obj: Dict[str, Any]) -> Tuple[str, bool, Optional[Dict[str, Any]], Optional[str]]:
    """Extract content piece and meta from a K2Think SSE JSON object."""
    if not isinstance(obj, dict):
        return "", False, None, None
    
    if obj.get("usage"):
        return "", False, obj.get("usage"), None
    
    if obj.get("done") is True:
        return "", True, obj.get("usage"), None
    
    if isinstance(obj.get("choices"), list) and obj["choices"]:
        delta = obj["choices"][0].get("delta") or {}
        role = delta.get("role")
        content_piece = delta.get("content") or ""
        return content_piece, False, None, role
    
    if isinstance(obj.get("content"), str):
        return obj["content"], False, None, None
    
    return "", False, None, None


async def stream_generator_k2_with_functions(
    payload: Dict[str, Any], 
    headers: Dict[str, str], 
    model: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
) -> AsyncGenerator[str, None]:
    """带函数调用支持的流式生成器"""
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())
    role_sent = False
    accumulated_content = ""
    previous_reasoning = ""
    previous_answer = ""
    reasoning_phase = True
    
    # 发送初始角色
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
                if not line or not line.startswith("data:"):
                    continue
                
                data_str = line[5:].strip()
                if not data_str or data_str == "-1":
                    continue
                if data_str in ("[DONE]", "DONE", "done"):
                    break

                content_piece: str = ""
                try:
                    obj = json.loads(data_str)
                    content_piece, is_done, _usage, role = _extract_content_from_json(obj)
                    if is_done:
                        break
                except Exception:
                    content_piece = data_str

                if content_piece:
                    accumulated_content = content_piece
                    
                    # 检查是否包含函数调用
                    if tools and "function_call" in accumulated_content.lower():
                        # 尝试解析函数调用
                        try:
                            # 这里需要根据实际响应格式调整解析逻辑
                            # 如果检测到函数调用，执行函数并返回结果
                            function_match = re.search(r'call_function\("([^"]+)"\s*,\s*({[^}]+})\)', accumulated_content)
                            if function_match:
                                func_name = function_match.group(1)
                                func_args = json.loads(function_match.group(2))
                                
                                # 执行函数调用
                                result = await execute_function_call(func_name, func_args)
                                
                                # 发送函数调用信息
                                tool_call = {
                                    "id": f"call_{uuid.uuid4().hex}",
                                    "type": "function",
                                    "function": {
                                        "name": func_name,
                                        "arguments": json.dumps(func_args)
                                    }
                                }
                                yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={'tool_calls': [tool_call]})]).json()}\n\n"
                                
                                # 发送函数结果
                                yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={'content': f'函数调用结果: {json.dumps(result, ensure_ascii=False)}'})]).json()}\n\n"
                                continue
                        except:
                            pass
                    
                    # 正常处理推理和回答内容
                    current_reasoning, current_answer = extract_reasoning_and_answer(accumulated_content)
                    
                    if reasoning_phase and current_reasoning:
                        reasoning_delta = calculate_delta_content(previous_reasoning, current_reasoning)
                        if reasoning_delta.strip():
                            yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={'reasoning_content': reasoning_delta})]).json()}\n\n"
                            previous_reasoning = current_reasoning
                    
                    if current_answer and reasoning_phase:
                        reasoning_phase = False
                        if current_reasoning and current_reasoning != previous_reasoning:
                            reasoning_delta = calculate_delta_content(previous_reasoning, current_reasoning)
                            if reasoning_delta.strip():
                                yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={'reasoning_content': reasoning_delta})]).json()}\n\n"
                    
                    if not reasoning_phase and current_answer:
                        answer_delta = calculate_delta_content(previous_answer, current_answer)
                        if answer_delta.strip():
                            yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={'content': answer_delta})]).json()}\n\n"
                            previous_answer = current_answer

    yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={}, finish_reason='stop')]).json()}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, 
    auth: Optional[HTTPAuthorizationCredentials] = Depends(authenticate_client)
):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages required")
    
    # 构建K2Think兼容的消息列表
    k2_messages: List[Dict[str, str]] = []
    system_prompt = ""
    
    for msg in request.messages:
        if msg.role == "system":
            system_prompt = msg.content if isinstance(msg.content, str) else convert_content_with_files(msg.content)
        elif msg.role in ("user", "assistant"):
            content = msg.content if isinstance(msg.content, str) else convert_content_with_files(msg.content)
            k2_messages.append({"role": msg.role, "content": content})
    
    # 如果有工具定义，添加到系统提示中
    if request.tools:
        tools_prompt = "\n\n可用工具:\n"
        for tool in request.tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                tools_prompt += f"- {func.get('name')}: {func.get('description')}\n"
                tools_prompt += f"  参数: {json.dumps(func.get('parameters', {}), ensure_ascii=False)}\n"
        
        system_prompt += tools_prompt
    
    if system_prompt:
        for m in k2_messages:
            if m.get("role") == "user":
                m["content"] = f"{system_prompt}\n\n{m['content']}"
                break
        else:
            k2_messages.insert(0, {"role": "user", "content": system_prompt})

    payload = {
        "stream": True,
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
        return StreamingResponse(
            stream_generator_k2_with_functions(
                payload, headers, request.model, request.tools, request.tool_choice
            ), 
            media_type="text/event-stream"
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail="K2Think API error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # 创建必要的配置文件
    if not os.path.exists("client_api_keys.json"):
        with open("client_api_keys.json", "w", encoding="utf-8") as f:
            json.dump([f"sk-talkai-{uuid.uuid4().hex}"], f)
    
    if not os.path.exists("functions.json"):
        functions_config = {
            "get_weather": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "温度单位"
                        }
                    },
                    "required": ["city"]
                }
            },
            "web_search": {
                "name": "web_search",
                "description": "搜索网络信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
        with open("functions.json", "w", encoding="utf-8") as f:
            json.dump(functions_config, f, indent=2, ensure_ascii=False)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
