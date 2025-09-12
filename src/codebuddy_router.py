"""
CodeBuddy API Router - 兼容CodeBuddy官方API格式
"""
import json
import time
import uuid
import secrets
import logging
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Request, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .auth import authenticate
from .codebuddy_api_client import codebuddy_api_client
from .codebuddy_token_manager import codebuddy_token_manager
from .usage_stats_manager import usage_stats_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# --- CodeBuddy V1 API Models ---

class CodeBuddyMessage(BaseModel):
    role: str
    content: Any  # 可以是字符串或复杂对象


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
        def apply_keyword_replacement(text):
            if isinstance(text, str):
                text = text.replace("Claude Code", "CodeBuddy Code")
                text = text.replace("Anthropic's official CLI for Claude", "Tencent's official CLI for CodeBuddy")
                text = text.replace("Claude", "CodeBuddy")
                text = text.replace("Anthropic", "Tencent")
                text = text.replace("https://github.com/anthropics/claude-code/issues", "https://cnb.cool/codebuddy/codebuddy-code/-/issues")
                return text
            return text
        
        # 对所有消息应用关键词替换
        for msg in payload.get("messages", []):
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                msg["content"] = apply_keyword_replacement(msg["content"])
            elif isinstance(msg.get("content"), str):
                msg["content"] = apply_keyword_replacement(msg["content"])
            elif isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        item["text"] = apply_keyword_replacement(item.get("text", ""))
        
        
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
                            error_chunk = f'data: {{"error": "CodeBuddy API error: {response.status_code} - {error_msg}"}}\n\n'
                            yield error_chunk.encode('utf-8')
                            return
                        
                        chunk_count = 0
                        total_bytes = 0
                        done_found = False
                        
                        # 使用 aiter_bytes 逐块读取，确保连接保持活跃
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            if chunk:
                                chunk_count += 1
                                total_bytes += len(chunk)
                                
                                
                                # 立即转发数据块给客户端
                                yield chunk
                                
                                # 检查是否包含结束标记
                                if b'[DONE]' in chunk:
                                    logger.info(f"[STREAM] 在chunk {chunk_count}中发现[DONE]标记")
                                    done_found = True
                                    # 不要在这里break，让流自然结束
                        
                        if not done_found:
                            logger.warning("[STREAM] 流结束但未发现[DONE]标记")
                                
                except httpx.TimeoutException as e:
                    logger.error(f"CodeBuddy API 超时: {e}")
                    error_chunk = f'data: {{"error": "CodeBuddy API timeout: {str(e)}"}}\n\n'
                    yield error_chunk.encode('utf-8')
                except httpx.NetworkError as e:
                    logger.error(f"网络错误: {e}")
                    error_chunk = f'data: {{"error": "Network error: {str(e)}"}}\n\n'
                    yield error_chunk.encode('utf-8')
                except Exception as e:
                    logger.error(f"流式响应错误: {e}", exc_info=True)
                    error_chunk = f'data: {{"error": "Stream interrupted: {str(e)}"}}\n\n'
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
                    logger.info("[AGG] enter non-stream aggregation")

                    async for _chunk in response.aiter_bytes():
                        if not _chunk:
                            continue
                        _s = _chunk.decode('utf-8', errors='ignore')
                        for _line in _s.split('\n'):
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
                                    if _pos is None or (_incoming_id and tool_calls[_pos].get('id') and tool_calls[_pos]['id'] != _incoming_id):
                                        tool_calls.append({
                                            'id': _incoming_id,
                                            'type': _tc.get('type', 'function'),
                                            'function': {'name': (_tc.get('function') or {}).get('name', ''), 'arguments': ''}
                                        })
                                        _pos = len(tool_calls) - 1
                                        active_pos_by_index[_idx] = _pos
                                        logger.info(f"[AGG] start tool_call index={_idx} id={_incoming_id} name={(_tc.get('function') or {}).get('name','')}")
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

            # Validate tool_calls.function.arguments JSON (log only, no mutation)
            try:
                if tool_calls:
                    for _i, _tc in enumerate(tool_calls):
                        _func = _tc.get('function') or {}
                        _args = _func.get('arguments')
                        if isinstance(_args, str) and _args.strip():
                            try:
                                json.loads(_args)
                            except Exception as _ve:
                                _preview = _args[:200].replace('\n', ' ')
                                logger.warning(f"[AGG_VALIDATE] tool_calls[{_i}] arguments not valid JSON: {_ve}; preview={_preview}")
                        else:
                            logger.debug(f"[AGG_VALIDATE] tool_calls[{_i}] has empty arguments")
            except Exception as _e:
                logger.warning(f"[AGG_VALIDATE] validation step error: {_e}")
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
            try:
                _tc_count = len(tool_calls)
            except Exception:
                _tc_count = 0
            logger.info(f"[AGG] finalize id={agg_id} model={agg_model} content_len={len(agg_content)} tool_calls={_tc_count} finish_reason={_finish_reason}")
            return _final_response
            
            # 收集所有流式响应块
            async for chunk in response.aiter_bytes():
                if chunk:
                    chunk_str = chunk.decode('utf-8')
                    for line in chunk_str.split('\n'):
                        if line.startswith('data: ') and not line.endswith('[DONE]'):
                            try:
                                data_str = line[6:]  # 移除 'data: ' 前缀
                                chunk_data = json.loads(data_str)
                                
                                # 收集所有有效的响应块
                                if "choices" in chunk_data:
                                    all_chunks.append(chunk_data)
                                    
                            except json.JSONDecodeError:
                                continue
            
            # 如果有响应块，合并为非流式格式
            if all_chunks:
                # 使用第一个块作为基础
                base_response = all_chunks[0].copy()
                base_response["object"] = "chat.completion"
                base_response["created"] = int(time.time())
                
                # 合并所有块的内容
                if "choices" in base_response and base_response["choices"]:
                    choice = base_response["choices"][0]
                    
                    # 如果第一个块就有完整的delta，直接转换
                    if "delta" in choice:
                        # 合并所有块的delta内容
                        merged_delta = choice["delta"].copy()
                        
                        
                        # 合并所有块的内容（包括第一个块）
                        merged_content = merged_delta.get("content", "") or ""
                        merged_tool_calls = {}  # 使用字典按index组织
                        
                        # 处理所有块（包括第一个）
                        for chunk in all_chunks:
                            if "choices" in chunk and chunk["choices"]:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta and delta["content"]:
                                    merged_content += delta["content"]
                                if "tool_calls" in delta:
                                    # 合并工具调用参数
                                    for new_tool_call in delta["tool_calls"]:
                                        if new_tool_call.get("index") is not None:
                                            index = new_tool_call["index"]
                                            
                                            # 初始化工具调用槽位
                                            if index not in merged_tool_calls:
                                                merged_tool_calls[index] = {
                                                    "index": index,
                                                    "function": {"name": "", "arguments": ""}
                                                }
                                            
                                            # 合并工具调用信息
                                            if "function" in new_tool_call:
                                                # 合并参数字符串
                                                if "arguments" in new_tool_call["function"]:
                                                    merged_tool_calls[index]["function"]["arguments"] += new_tool_call["function"]["arguments"]
                                                
                                                # 设置工具名称（保留非空值）
                                                if "name" in new_tool_call["function"] and new_tool_call["function"]["name"]:
                                                    merged_tool_calls[index]["function"]["name"] = new_tool_call["function"]["name"]
                                            
                                            # 更新其他字段（id, type等）
                                            for key, value in new_tool_call.items():
                                                if key not in ["function", "index"]:
                                                    merged_tool_calls[index][key] = value
                        
                        # 转换为数组格式
                        merged_delta["content"] = merged_content
                        if merged_tool_calls:
                            merged_delta["tool_calls"] = list(merged_tool_calls.values())
                        
                        # 转换delta为message
                        choice["message"] = merged_delta
                        choice["message"]["role"] = choice["message"].get("role", "assistant")
                        choice["finish_reason"] = "stop"
                        if 'delta' in choice:
                            del choice["delta"]
                        
                
                return base_response
            else:
                # 如果没有收到有效响应，返回错误
                return {
                    "error": "No valid response received from CodeBuddy",
                    "details": "Stream ended without complete response"
                }
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CodeBuddy V1 API错误: {e}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@router.get("/v1/models")
async def list_v1_models(_token: str = Depends(authenticate)):
    """获取CodeBuddy V1模型列表"""
    try:
        # NOTE: 此端点不再消耗凭证，因为它只返回一个静态列表。
        # 只需要服务级别的认证即可。
        # credential = codebuddy_token_manager.get_next_credential()
        # if not credential:
        #     raise HTTPException(status_code=401, detail="没有可用的CodeBuddy凭证")
        
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
