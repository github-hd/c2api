"""
CodeBuddy API Router - 兼容CodeBuddy官方API格式
"""
import json
import time
import uuid
import logging
import asyncio
import httpx
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Request, Header
from fastapi.responses import StreamingResponse

from .auth import authenticate
from .codebuddy_api_client import codebuddy_api_client
from .codebuddy_token_manager import codebuddy_token_manager
from .usage_stats_manager import usage_stats_manager
from .keyword_replacer import apply_keyword_replacement_to_system_message

logger = logging.getLogger(__name__)

router = APIRouter()

# --- 辅助函数 ---

def format_sse_error(message: str, error_type: str = "stream_error") -> str:
    """格式化SSE错误响应"""
    error_data = {
        "error": {
            "message": message,
            "type": error_type
        }
    }
    return f'data: {json.dumps(error_data, ensure_ascii=False)}\n\n'

def parse_sse_line(line: str) -> Optional[Dict[str, Any]]:
    """解析单行SSE数据"""
    if not line.startswith('data: '):
        return None
    
    data = line[6:].strip()
    if not data or data == '[DONE]':
        return None
    
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None

def validate_and_fix_tool_call_args(args: str) -> str:
    """验证和修复工具调用参数的JSON格式"""
    if not args:
        return '{}'
    
    try:
        json.loads(args)
        return args
    except json.JSONDecodeError:
        # 尝试修复常见的JSON问题
        if not args.endswith('}') and args.count('{') > args.count('}'):
            args += '}'
        elif not args.endswith(']') and args.count('[') > args.count(']'):
            args += ']'
        
        try:
            json.loads(args)
            return args
        except:
            logger.warning(f"无法修复工具调用参数JSON: {args}")
            return '{}'

class SSEConnectionManager:
    """SSE 连接管理器，包含重连逻辑"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    async def stream_with_retry(self, stream_func, *args, **kwargs):
        """带重连的流式处理"""
        for attempt in range(self.max_retries + 1):
            try:
                async for chunk in stream_func(*args, **kwargs):
                    yield chunk
                break  # 成功完成，退出重试循环
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)  # 指数退避: 1s, 2s, 4s
                    logger.warning(f"连接失败，{wait_time}秒后重试 (第{attempt + 1}次): {e}")
                    yield format_sse_error(f"Connection lost, retrying in {wait_time}s... (attempt {attempt + 1})", "connection_retry")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"重连失败，已达到最大重试次数: {e}")
                    yield format_sse_error(f"Connection failed after {self.max_retries} retries: {str(e)}", "connection_failed")
                    raise
            except Exception as e:
                # 其他异常不重试，直接抛出
                logger.error(f"流式处理异常: {e}")
                yield format_sse_error(f"Stream error: {str(e)}", "stream_error")
                raise

class StreamResponseAggregator:
    """流式响应聚合器"""
    
    def __init__(self):
        self.data = {
            "id": None,
            "model": None,
            "content": "",
            "tool_calls": [],
            "finish_reason": None,
            "usage": None,
            "system_fingerprint": None
        }
        self.tool_call_map = {}
    
    def process_chunk(self, obj: Dict[str, Any]):
        """处理单个响应块"""
        # 聚合基本信息
        self.data["id"] = self.data["id"] or obj.get('id')
        self.data["model"] = self.data["model"] or obj.get('model')
        self.data["system_fingerprint"] = obj.get('system_fingerprint') or self.data["system_fingerprint"]
        
        if obj.get('usage'):
            self.data["usage"] = obj.get('usage')
        
        choices = obj.get('choices', [])
        if not choices:
            return
        
        choice = choices[0]
        if choice.get('finish_reason'):
            self.data["finish_reason"] = choice.get('finish_reason')
        
        delta = choice.get('delta', {})
        
        # 聚合内容
        if delta.get('content'):
            self.data["content"] += delta.get('content')
        
        # 处理工具调用
        if delta.get('tool_calls'):
            self._process_tool_calls(delta.get('tool_calls'))
    
    def _process_tool_calls(self, tool_calls: List[Dict[str, Any]]):
        """处理工具调用"""
        for tc in tool_calls:
            idx = tc.get('index')
            if idx is None:
                continue
            
            # 初始化工具调用
            if idx not in self.tool_call_map:
                self.tool_call_map[idx] = {
                    'id': tc.get('id', ''),
                    'type': tc.get('type', 'function'),
                    'function': {
                        'name': '',
                        'arguments': ''
                    }
                }
            
            # 更新工具调用信息
            if tc.get('id'):
                self.tool_call_map[idx]['id'] = tc.get('id')
            if tc.get('type'):
                self.tool_call_map[idx]['type'] = tc.get('type')
            
            func = tc.get('function', {})
            if func.get('name'):
                self.tool_call_map[idx]['function']['name'] = func.get('name')
            if func.get('arguments'):
                self.tool_call_map[idx]['function']['arguments'] += func.get('arguments')
    
    def finalize(self) -> Dict[str, Any]:
        """完成聚合并返回最终响应"""
        # 构建工具调用列表
        if self.tool_call_map:
            self.data["tool_calls"] = [self.tool_call_map[i] for i in sorted(self.tool_call_map.keys())]
            
            # 验证和修复工具调用参数
            for tc in self.data["tool_calls"]:
                tc['function']['arguments'] = validate_and_fix_tool_call_args(
                    tc['function']['arguments']
                )
        
        # 构建最终响应
        final_message = {"role": "assistant", "content": self.data["content"]}
        if self.data["tool_calls"]:
            final_message["tool_calls"] = self.data["tool_calls"]
        
        finish_reason = "tool_calls" if self.data["tool_calls"] else (self.data["finish_reason"] or "stop")
        
        final_response = {
            "id": self.data["id"] or str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.data["model"] or "unknown",
            "choices": [
                {
                    "index": 0,
                    "message": final_message,
                    "finish_reason": finish_reason,
                    "logprobs": None
                }
            ]
        }
        
        if self.data["usage"]:
            final_response["usage"] = self.data["usage"]
        if self.data["system_fingerprint"]:
            final_response["system_fingerprint"] = self.data["system_fingerprint"]
        
        return final_response

# --- API Endpoints ---

@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,  # 使用原始 Request 对象，绕过 Pydantic 验证
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_conversation_request_id: Optional[str] = Header(None, alias="X-Conversation-Request-ID"),
    x_conversation_message_id: Optional[str] = Header(None, alias="X-Conversation-Message-ID"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    _token: str = Depends(authenticate)
):
    """
    CodeBuddy V1 聊天完成API - 完全透传模式
    """
    try:
        # 获取原始请求体
        try:
            request_body = await request.json()
        except Exception as e:
            logger.error(f"解析请求体失败: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON request body: {str(e)}")
        
        # 获取CodeBuddy凭证
        credential = codebuddy_token_manager.get_next_credential()
        if not credential:
            raise HTTPException(status_code=401, detail="没有可用的CodeBuddy凭证")
        
        bearer_token = credential.get('bearer_token')
        user_id = credential.get('user_id')
        
        if not bearer_token:
            raise HTTPException(status_code=401, detail="无效的CodeBuddy凭证")
        
        # 生成请求头
        headers = codebuddy_api_client.generate_codebuddy_headers(
            bearer_token=bearer_token,
            user_id=user_id,
            conversation_id=x_conversation_id,
            conversation_request_id=x_conversation_request_id,
            conversation_message_id=x_conversation_message_id,
            request_id=x_request_id
        )
        
        # 完全透传请求体，但需要处理一些 CodeBuddy 的特殊要求
        payload = request_body.copy()
        
        # Record model usage stats
        model_name = payload.get("model", "unknown")
        usage_stats_manager.record_model_usage(model_name)
        
        payload["stream"] = True  # CodeBuddy 只支持流式请求
        
        # 处理消息长度要求：CodeBuddy要求至少2条消息
        messages = payload.get("messages", [])
        if len(messages) == 1 and messages[0].get("role") == "user":
            # 添加系统消息
            system_msg = {
                "role": "system",
                "content": "You are a helpful assistant."
            }
            payload["messages"] = [system_msg] + messages
        
        # 应用关键词替换 - 防止CodeBuddy检测到竞争对手关键词
        # 只对 system role 的消息应用关键词替换
        for msg in payload.get("messages", []):
            if msg.get("role") == "system":
                msg["content"] = apply_keyword_replacement_to_system_message(msg.get("content"))
        
        # 检查客户端是否期望流式响应
        client_wants_stream = request_body.get("stream", False)
        
        # 发送请求到CodeBuddy
        
        if client_wants_stream:
            # 带重连机制的流式响应
            async def stream_response_core():
                """核心流式响应函数"""
                # 使用标准的客户端配置和上下文管理
                async with httpx.AsyncClient(
                    verify=False,
                    timeout=httpx.Timeout(300.0, connect=30.0, read=300.0)
                ) as client:
                    
                    # 使用stream方法确保连接保持活跃
                    async with client.stream(
                        "POST",
                        "https://www.codebuddy.ai/v2/chat/completions",
                        json=payload,
                        headers=headers
                    ) as response:
                        
                        if response.status_code != 200:
                            error_text = await response.aread()
                            error_msg = error_text.decode('utf-8', errors='ignore')
                            logger.error(f"CodeBuddy API错误: {response.status_code} - {error_msg}")
                            yield format_sse_error(f"CodeBuddy API error: {response.status_code} - {error_msg}", "api_error")
                            return
                        
                        # 简化的SSE处理：按行解析
                        buffer = ""
                        async for chunk in response.aiter_text(chunk_size=8192):
                            if not chunk:
                                continue
                            
                            buffer += chunk
                            
                            # 按行处理，确保SSE事件完整性
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                
                                # 跳过空行和注释
                                if not line.strip() or line.startswith(':'):
                                    continue
                                
                                # 直接转发完整的SSE行
                                yield line + '\n'
                                
                                # 检查结束标记
                                if '[DONE]' in line:
                                    return
                        
                        # 处理缓冲区中剩余的数据
                        if buffer.strip():
                            yield buffer + '\n'
            
            # 使用重连管理器
            connection_manager = SSEConnectionManager(max_retries=3, retry_delay=1.0)
            
            async def stream_response():
                async for chunk in connection_manager.stream_with_retry(stream_response_core):
                    yield chunk
            
            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "*"
                }
            )
        else:
            # 简化的非流式响应聚合
            try:
                async with httpx.AsyncClient(verify=False, timeout=300) as client:
                    response = await client.post(
                        "https://www.codebuddy.ai/v2/chat/completions",
                        json=payload,
                        headers=headers
                    )
                    
                    if response.status_code != 200:
                        error_text = response.text
                        logger.error(f"CodeBuddy API错误: {response.status_code} - {error_text}")
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"CodeBuddy API error: {error_text}"
                        )
                    
                    # 使用聚合器处理响应
                    aggregator = StreamResponseAggregator()
                    
                    # 简化的SSE解析
                    buffer = ""
                    async for chunk in response.aiter_text():
                        if not chunk:
                            continue
                        
                        buffer += chunk
                        
                        # 按行处理SSE事件
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            
                            obj = parse_sse_line(line)
                            if obj:
                                aggregator.process_chunk(obj)
                    
                    # 处理缓冲区剩余数据
                    if buffer.strip():
                        obj = parse_sse_line(buffer.strip())
                        if obj:
                            aggregator.process_chunk(obj)

            except httpx.TimeoutException:
                logger.error("CodeBuddy API 超时")
                raise HTTPException(status_code=504, detail="CodeBuddy API timeout")
            except httpx.NetworkError as e:
                logger.error(f"网络错误: {e}")
                raise HTTPException(status_code=502, detail=f"Network error: {str(e)}")
            except Exception as e:
                logger.error(f"请求异常: {e}")
                raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")

            # 返回聚合后的最终响应
            return aggregator.finalize()
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CodeBuddy V1 API错误: {e}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@router.get("/v1/models")
async def list_v1_models(_token: str = Depends(authenticate)):
    """获取CodeBuddy V1模型列表"""
    try:
        models = [
            "claude-4.0", "claude-3.7", "gpt-5", "gpt-5-mini",
            "gpt-5-nano", "o4-mini", "gemini-2.5-flash",
            "gemini-2.5-pro", "auto-chat"
        ]
        
        return {
            "object": "list",
            "data": [{
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "codebuddy"
            } for model in models]
        }
        
    except Exception as e:
        logger.error(f"获取V1模型列表错误: {e}")
        raise HTTPException(status_code=500, detail="获取模型列表失败")

@router.get("/v1/credentials", summary="List all available credentials")
async def list_credentials(_token: str = Depends(authenticate)):
    """列出所有可用凭证的详细信息，包括过期状态"""
    try:
        credentials_info = codebuddy_token_manager.get_credentials_info()
        safe_credentials = []
        
        credentials = codebuddy_token_manager.get_all_credentials()
        
        for info in credentials_info:
            bearer_token = credentials[info['index']].get("bearer_token", "") if info['index'] < len(credentials) else ""
            
            # 格式化时间显示
            if info['time_remaining'] is not None and info['time_remaining'] > 0:
                days, remainder = divmod(info['time_remaining'], 86400)
                hours, remainder = divmod(remainder, 3600)
                minutes = remainder // 60
                time_remaining_str = f"{days}d {hours}h" if days > 0 else f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
            else:
                time_remaining_str = "Expired" if info['time_remaining'] is not None else "Unknown"
            
            safe_credentials.append({
                **info,  # 展开所有原始信息
                "time_remaining_str": time_remaining_str,
                "has_token": bool(bearer_token),
                "token_preview": f"{bearer_token[:10]}...{bearer_token[-4:]}" if len(bearer_token) > 14 else "Invalid Token"
            })
        
        return {"credentials": safe_credentials}
        
    except Exception as e:
        logger.error(f"获取凭证列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/credentials", summary="Add a new credential")
async def add_credential(
    request: Request,
    _token: str = Depends(authenticate)
):
    """添加一个新的认证凭证"""
    try:
        data = await request.json()
        if not data.get("bearer_token"):
            raise HTTPException(status_code=422, detail="bearer_token is required")

        success = codebuddy_token_manager.add_credential(
            data.get("bearer_token"),
            data.get("user_id"),
            data.get("filename")
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save credential file")
        
        return {"message": "Credential added successfully"}

    except Exception as e:
        logger.error(f"添加凭证失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/credentials/select", summary="Manually select a credential")
async def select_credential(
    request: Request,
    _token: str = Depends(authenticate)
):
    """手动选择指定的凭证"""
    try:
        data = await request.json()
        index = data.get("index")
        if index is None:
            raise HTTPException(status_code=422, detail="index is required")

        if not codebuddy_token_manager.set_manual_credential(index):
            raise HTTPException(status_code=400, detail="Invalid credential index")
        
        return {"message": f"Credential #{index + 1} selected successfully"}

    except Exception as e:
        logger.error(f"选择凭证失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/credentials/auto", summary="Resume automatic credential rotation")
async def resume_auto_rotation(_token: str = Depends(authenticate)):
    """恢复自动凭证轮换"""
    try:
        codebuddy_token_manager.clear_manual_selection()
        return {"message": "Resumed automatic credential rotation"}

    except Exception as e:
        logger.error(f"恢复自动轮换失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/credentials/toggle-rotation", summary="Toggle automatic credential rotation")
async def toggle_auto_rotation(_token: str = Depends(authenticate)):
    """切换自动轮换开关"""
    try:
        is_enabled = codebuddy_token_manager.toggle_auto_rotation()
        status = "enabled" if is_enabled else "disabled"
        message = f"Auto rotation {status}"
        return {
            "message": message,
            "auto_rotation_enabled": is_enabled
        }

    except Exception as e:
        logger.error(f"切换自动轮换失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/credentials/current", summary="Get current credential info")
async def get_current_credential(_token: str = Depends(authenticate)):
    """获取当前使用的凭证信息"""
    try:
        info = codebuddy_token_manager.get_current_credential_info()
        return info

    except Exception as e:
        logger.error(f"获取当前凭证信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/credentials/delete", summary="Delete a credential by index")
async def delete_credential(request: Request, _token: str = Depends(authenticate)):
    """删除一个凭证文件（通过索引）并从列表中移除"""
    try:
        data = await request.json()
        index = data.get("index")
        if index is None or not isinstance(index, int):
            raise HTTPException(status_code=422, detail="Valid integer index is required")

        if not codebuddy_token_manager.delete_credential_by_index(index):
            raise HTTPException(status_code=400, detail="Invalid index or failed to delete credential")

        return {"message": f"Credential #{index + 1} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除凭证失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
