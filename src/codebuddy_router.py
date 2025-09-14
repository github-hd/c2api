"""
CodeBuddy API Router - 兼容CodeBuddy官方API格式
"""
import json
import time
import uuid
import logging
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
        import httpx
        
        if client_wants_stream:
            # 流式响应：使用httpx.stream确保连接保持活跃
            async def stream_response():
                client = None
                try:
                    # 创建客户端连接，使用合适的超时设置
                    client = httpx.AsyncClient(
                        verify=False, 
                        timeout=httpx.Timeout(300.0, connect=30.0, read=300.0)
                    )
                    
                    # 使用stream方法确保连接在整个过程中保持活跃
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
                            error_data = {"error": f"CodeBuddy API error: {response.status_code} - {error_msg}"}
                            error_chunk = f'data: {json.dumps(error_data, ensure_ascii=False)}\n\n'
                            yield error_chunk.encode('utf-8')
                            return
                        
                        chunk_count = 0
                        total_bytes = 0
                        done_found = False
                        
                        # 使用缓冲区逐行处理，确保SSE事件完整性
                        buffer = b""
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            if chunk:
                                chunk_count += 1
                                total_bytes += len(chunk)
                                
                                # 将chunk添加到缓冲区
                                buffer += chunk
                                
                                # 按行分割处理，确保SSE事件的完整性
                                while b'\n' in buffer:
                                    line, buffer = buffer.split(b'\n', 1)
                                    line_with_newline = line + b'\n'
                                    
                                    # 立即转发完整的行给客户端
                                    yield line_with_newline
                                    
                                    # 检查是否包含结束标记
                                    if b'[DONE]' in line:
                                        done_found = True
                        
                        # 处理缓冲区中剩余的数据（确保SSE事件完整性）
                        if buffer:
                            # 检查是否是完整的SSE事件
                            buffer_str = buffer.decode('utf-8', errors='ignore')
                            if buffer_str.strip():
                                # 如果不是以换行符结尾，添加换行符确保SSE格式正确
                                if not buffer_str.endswith('\n'):
                                    buffer += b'\n'
                                yield buffer
                        
                        if not done_found:
                            logger.warning("[STREAM] 流结束但未发现[DONE]标记")
                                
                except httpx.TimeoutException as e:
                    logger.error(f"CodeBuddy API 超时: {e}")
                    error_data = {"error": f"CodeBuddy API timeout: {str(e)}"}
                    error_chunk = f'data: {json.dumps(error_data, ensure_ascii=False)}\n\n'
                    yield error_chunk.encode('utf-8')
                except httpx.NetworkError as e:
                    logger.error(f"网络错误: {e}")
                    error_data = {"error": f"Network error: {str(e)}"}
                    error_chunk = f'data: {json.dumps(error_data, ensure_ascii=False)}\n\n'
                    yield error_chunk.encode('utf-8')
                except Exception as e:
                    logger.error(f"流式响应错误: {e}", exc_info=True)
                    error_data = {"error": f"Stream interrupted: {str(e)}"}
                    error_chunk = f'data: {json.dumps(error_data, ensure_ascii=False)}\n\n'
                    yield error_chunk.encode('utf-8')
                finally:
                    # 确保客户端连接被正确关闭
                    if client:
                        await client.aclose()
            
            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                    "Content-Type": "text/event-stream; charset=utf-8",
                    "Transfer-Encoding": "chunked"
                }
            )
        else:
            # 客户端要求非流式，收集并合并所有流式响应块（真正透传）
            # Rewritten: aggregate SSE chunks into a single non-stream response (minimal + complete)
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
                    
                    agg_id = None
                    agg_model = None
                    agg_content = ""
                    # Maintain ordered list of tool calls + active mapping by index
                    tool_calls: List[Dict[str, Any]] = []
                    active_pos_by_index: Dict[int, int] = {}
                    last_finish_reason = None
                    last_usage = None
                    last_system_fp = None
                    saw_any = False

                    # 使用缓冲区确保SSE事件完整性（非流式也需要）
                    buffer = b""
                    async for _chunk in response.aiter_bytes():
                        if not _chunk:
                            continue
                        buffer += _chunk
                        
                        # 按行处理，确保完整的SSE事件
                        while b'\n' in buffer:
                            line_bytes, buffer = buffer.split(b'\n', 1)
                            _line = line_bytes.decode('utf-8', errors='ignore')
                            
                            if not _line.startswith('data: '):
                                continue
                            _data = _line[6:].strip()
                            if not _data or _data == '[DONE]':
                                continue
                            try:
                                _obj = json.loads(_data)
                            except json.JSONDecodeError:
                                continue

                            saw_any = True
                            agg_id = agg_id or _obj.get('id')
                            agg_model = agg_model or _obj.get('model')
                            last_system_fp = _obj.get('system_fingerprint') or last_system_fp
                            if _obj.get('usage'):
                                last_usage = _obj.get('usage')

                            _choices = _obj.get('choices') or []
                            if not _choices:
                                continue
                            _c0 = _choices[0]
                            _fr = _c0.get('finish_reason')
                            if _fr:
                                last_finish_reason = _fr
                            _delta = _c0.get('delta') or {}
                            _text_piece = _delta.get('content')
                            if _text_piece:
                                agg_content += _text_piece
                            _tc_list = _delta.get('tool_calls')
                            if isinstance(_tc_list, list):
                                for _tc in _tc_list:
                                    if not isinstance(_tc, dict):
                                        continue
                                    _idx = _tc.get('index')
                                    if _idx is None:
                                        continue
                                    _incoming_id = _tc.get('id')
                                    _pos = active_pos_by_index.get(_idx)
                                    # Start a new call when first seen for this index or id changes
                                    if _pos is None or (_pos is not None and _incoming_id and _pos < len(tool_calls) and tool_calls[_pos].get('id') and tool_calls[_pos]['id'] != _incoming_id):
                                        tool_calls.append({
                                            'id': _incoming_id,
                                            'type': _tc.get('type', 'function'),
                                            'function': {'name': (_tc.get('function') or {}).get('name', ''), 'arguments': ''}
                                        })
                                        _pos = len(tool_calls) - 1
                                        active_pos_by_index[_idx] = _pos
                                    # Merge updates into active call
                                    if _incoming_id and not tool_calls[_pos].get('id'):
                                        tool_calls[_pos]['id'] = _incoming_id
                                    if _tc.get('type'):
                                        tool_calls[_pos]['type'] = _tc.get('type')
                                    _func = _tc.get('function') or {}
                                    _name = _func.get('name')
                                    if _name:
                                        tool_calls[_pos]['function']['name'] = _name
                                    _args_piece = _func.get('arguments')
                                    if _args_piece:
                                        tool_calls[_pos]['function']['arguments'] += _args_piece

                    # 处理缓冲区中剩余的最后一行数据（如果存在）
                    if buffer:
                        line_bytes = buffer
                        _line = line_bytes.decode('utf-8', errors='ignore').strip()
                        
                        if _line.startswith('data: '):
                            _data = _line[6:].strip()
                            if _data and _data != '[DONE]':
                                try:
                                    _obj = json.loads(_data)
                                    # 安全地处理最后一行数据，使用相同的逻辑
                                    saw_any = True
                                    agg_id = agg_id or _obj.get('id')
                                    agg_model = agg_model or _obj.get('model')
                                    last_system_fp = _obj.get('system_fingerprint') or last_system_fp
                                    if _obj.get('usage'):
                                        last_usage = _obj.get('usage')

                                    _choices = _obj.get('choices') or []
                                    if _choices:
                                        _c0 = _choices[0]
                                        _fr = _c0.get('finish_reason')
                                        if _fr:
                                            last_finish_reason = _fr
                                        _delta = _c0.get('delta') or {}
                                        _text_piece = _delta.get('content')
                                        if _text_piece:
                                            agg_content += _text_piece
                                        # 处理工具调用（如果有）
                                        _tc_list = _delta.get('tool_calls')
                                        if isinstance(_tc_list, list):
                                            for _tc in _tc_list:
                                                if not isinstance(_tc, dict):
                                                    continue
                                                _idx = _tc.get('index')
                                                if _idx is None:
                                                    continue
                                                _incoming_id = _tc.get('id')
                                                _pos = active_pos_by_index.get(_idx)
                                                if _pos is None or (_pos is not None and _incoming_id and _pos < len(tool_calls) and tool_calls[_pos].get('id') and tool_calls[_pos]['id'] != _incoming_id):
                                                    tool_calls.append({
                                                        'id': _incoming_id,
                                                        'type': _tc.get('type', 'function'),
                                                        'function': {'name': (_tc.get('function') or {}).get('name', ''), 'arguments': ''}
                                                    })
                                                    _pos = len(tool_calls) - 1
                                                    active_pos_by_index[_idx] = _pos
                                                if _pos < len(tool_calls):  # 安全检查
                                                    if _incoming_id and not tool_calls[_pos].get('id'):
                                                        tool_calls[_pos]['id'] = _incoming_id
                                                    if _tc.get('type'):
                                                        tool_calls[_pos]['type'] = _tc.get('type')
                                                    _func = _tc.get('function') or {}
                                                    _name = _func.get('name')
                                                    if _name:
                                                        tool_calls[_pos]['function']['name'] = _name
                                                    _args_piece = _func.get('arguments')
                                                    if _args_piece:
                                                        tool_calls[_pos]['function']['arguments'] += _args_piece
                                except json.JSONDecodeError:
                                    # 静默忽略无效的JSON，避免破坏现有逻辑
                                    pass

            except httpx.TimeoutException:
                logger.error("CodeBuddy API 超时")
                raise HTTPException(status_code=504, detail="CodeBuddy API timeout")
            except httpx.NetworkError as e:
                logger.error(f"网络错误: {e}")
                raise HTTPException(status_code=502, detail=f"Network error: {str(e)}")
            except Exception as e:
                logger.error(f"请求异常: {e}")
                raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")

            if not saw_any:
                return {"error": "No valid response received from CodeBuddy", "details": "Stream ended without complete response"}

            _final_message = {"role": "assistant", "content": agg_content}
            if tool_calls:
                _final_message["tool_calls"] = tool_calls
            _finish_reason = "tool_calls" if tool_calls else (last_finish_reason or "stop")

            # Validate and fix tool_calls.function.arguments JSON
            if tool_calls:
                for _i, _tc in enumerate(tool_calls):
                    _func = _tc.get('function') or {}
                    _args = _func.get('arguments', '')
                    if isinstance(_args, str) and _args.strip():
                        try:
                            # 尝试解析JSON以验证格式
                            json.loads(_args)
                        except json.JSONDecodeError as _ve:
                            # 如果JSON无效，尝试修复常见问题
                            logger.warning(f"Tool call {_i} has invalid JSON arguments: {_ve}")
                            
                            # 尝试修复截断的JSON
                            if not _args.endswith('}') and not _args.endswith(']'):
                                if _args.count('{') > _args.count('}'):
                                    _args += '}'
                                elif _args.count('[') > _args.count(']'):
                                    _args += ']'
                            
                            # 再次尝试解析
                            try:
                                json.loads(_args)
                                _tc['function']['arguments'] = _args
                                logger.info(f"Successfully fixed tool call {_i} JSON")
                            except:
                                # 如果仍然无法解析，使用空对象
                                logger.error(f"Could not fix tool call {_i}, using empty args")
                                _tc['function']['arguments'] = '{}'
            _final_response = {
                "id": agg_id or str(uuid.uuid4()),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": agg_model or "unknown",
                "choices": [
                    {
                        "index": 0,
                        "message": _final_message,
                        "finish_reason": _finish_reason,
                        "logprobs": None
                    }
                ]
            }
            if last_usage:
                _final_response["usage"] = last_usage
            if last_system_fp:
                _final_response["system_fingerprint"] = last_system_fp
            return _final_response
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CodeBuddy V1 API错误: {e}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@router.get("/v1/models")
async def list_v1_models(_token: str = Depends(authenticate)):
    """获取CodeBuddy V1模型列表"""
    try:
        # 此端点只返回静态模型列表，不需要消耗凭证
        
        models = [
            "claude-4.0",
            "claude-3.7",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "o4-mini",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "auto-chat"
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
        
        for info in credentials_info:
            bearer_token = ""
            # 从原始数据中获取token用于预览
            credentials = codebuddy_token_manager.get_all_credentials()
            if info['index'] < len(credentials):
                bearer_token = credentials[info['index']].get("bearer_token", "")
            
            # 格式化时间显示
            time_remaining_str = "Unknown"
            if info['time_remaining'] is not None:
                if info['time_remaining'] > 0:
                    days = info['time_remaining'] // 86400
                    hours = (info['time_remaining'] % 86400) // 3600
                    minutes = (info['time_remaining'] % 3600) // 60
                    if days > 0:
                        time_remaining_str = f"{days}d {hours}h"
                    elif hours > 0:
                        time_remaining_str = f"{hours}h {minutes}m"
                    else:
                        time_remaining_str = f"{minutes}m"
                else:
                    time_remaining_str = "Expired"
            
            safe_cred = {
                "index": info['index'],
                "filename": info['filename'],
                "user_id": info['user_id'],
                "email": info['email'],
                "name": info['name'],
                "created_at": info['created_at'],
                "expires_in": info['expires_in'],
                "expires_at": info['expires_at'],
                "time_remaining": info['time_remaining'],
                "time_remaining_str": time_remaining_str,
                "is_expired": info['is_expired'],
                "token_type": info['token_type'],
                "scope": info['scope'],
                "domain": info['domain'],
                "has_refresh_token": info['has_refresh_token'],
                "session_state": info['session_state'],
                "has_token": bool(bearer_token),
                "token_preview": f"{bearer_token[:10]}...{bearer_token[-4:]}" if bearer_token and len(bearer_token) > 14 else "Invalid Token"
            }
            safe_credentials.append(safe_cred)
        
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
        bearer_token = data.get("bearer_token")
        user_id = data.get("user_id")
        filename = data.get("filename")

        if not bearer_token:
            raise HTTPException(status_code=422, detail="bearer_token is required")

        success = codebuddy_token_manager.add_credential(bearer_token, user_id, filename)
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

        success = codebuddy_token_manager.set_manual_credential(index)
        if not success:
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

        if index is None:
            raise HTTPException(status_code=422, detail="index is required")

        if not isinstance(index, int):
            raise HTTPException(status_code=422, detail="index must be an integer")

        success = codebuddy_token_manager.delete_credential_by_index(index)
        if not success:
            raise HTTPException(status_code=400, detail="Invalid index or failed to delete credential")

        return {"message": f"Credential #{index + 1} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除凭证失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
