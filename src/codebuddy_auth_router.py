"""
CodeBuddy Authentication Router
基于真实CodeBuddy API的认证实现
"""
import hashlib
import secrets
import httpx
import base64
import json
import uuid
import time
from typing import Dict, Any, Optional
from fastapi.responses import JSONResponse
from fastapi import APIRouter, HTTPException, Depends, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import get_server_password
import logging

logger = logging.getLogger(__name__)

# --- Constants ---
CODEBUDDY_BASE_URL = 'https://www.codebuddy.ai'
CODEBUDDY_AUTH_TOKEN_ENDPOINT = f'{CODEBUDDY_BASE_URL}/v2/plugin/auth/token'
CODEBUDDY_AUTH_STATE_ENDPOINT = f'{CODEBUDDY_BASE_URL}/v2/plugin/auth/state'

# --- Router Setup ---
router = APIRouter()
security = HTTPBearer()

# --- JWT Authentication ---
import jwt

def get_jwt_secret():
    """基于服务密码生成JWT密钥"""
    password = get_server_password()
    if not password:
        return "fallback-secret-for-development-only"
    return hashlib.sha256(password.encode()).hexdigest()

JWT_SECRET = get_jwt_secret()
ALGORITHM = "HS256"

def authenticate(credentials = Depends(security)) -> str:
    """基于服务密码的认证"""
    password = get_server_password()
    if not password:
        raise HTTPException(status_code=500, detail="CODEBUDDY_PASSWORD is not configured on the server.")
    
    token = credentials.credentials
    if token != password:
        raise HTTPException(status_code=403, detail="Invalid password")
    return token

# --- Helper Functions ---
def generate_auth_state() -> str:
    """生成CodeBuddy认证的state参数"""
    timestamp = int(time.time())
    random_part = secrets.token_hex(16)
    return f"{random_part}_{timestamp}"

def get_codebuddy_headers() -> Dict[str, str]:
    """获取CodeBuddy API所需的标准请求头"""
    request_id = str(uuid.uuid4()).replace('-', '')
    return {
        'Accept': 'application/json, text/plain, */*',
        'X-Requested-With': 'XMLHttpRequest',
        'X-Request-ID': request_id,
        'b3': f'{request_id}-{secrets.token_hex(8)}-1-',
        'X-B3-TraceId': request_id,
        'X-B3-ParentSpanId': '',
        'X-B3-SpanId': secrets.token_hex(8),
        'X-B3-Sampled': '1',
        'X-No-Authorization': 'true',
        'X-No-User-Id': 'true',
        'X-No-Enterprise-Id': 'true',
        'X-No-Department-Info': 'true',
        'X-Domain': 'www.codebuddy.ai',
        'User-Agent': 'CLI/1.0.7 CodeBuddy/1.0.7',
        'X-Product': 'SaaS',
        'Host': 'www.codebuddy.ai'
    }

async def start_codebuddy_auth() -> Dict[str, Any]:
    """启动CodeBuddy认证流程"""
    try:
        logger.info("启动CodeBuddy认证流程...")
        
        headers = get_codebuddy_headers()
        
        # 调用 /v2/plugin/auth/state 获取认证状态和URL
        async with httpx.AsyncClient(verify=False) as client:
            state_url = f"{CODEBUDDY_AUTH_STATE_ENDPOINT}?platform=CLI"
            payload = {}
            
            response = await client.post(state_url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == 0 and result.get('data'):
                    data = result['data']
                    auth_state = data.get('state')
                    auth_url = data.get('authUrl')
                    
                    if auth_state and auth_url:
                        token_endpoint = f"{CODEBUDDY_AUTH_TOKEN_ENDPOINT}?state={auth_state}"
                        
                        return {
                            "success": True,
                            "method": "codebuddy_real_auth",
                            "auth_state": auth_state,
                            "verification_uri_complete": auth_url,
                            "verification_uri": CODEBUDDY_BASE_URL,
                            "token_endpoint": token_endpoint,
                            "expires_in": 1800,
                            "interval": 5,
                            "status": "awaiting_login",
                            "instructions": "请点击链接完成CodeBuddy登录",
                            "message": "请使用提供的链接登录CodeBuddy",
                            "platform": "CLI"
                        }
                        
        return {
            "success": False,
            "error": "auth_start_failed",
            "message": "无法启动认证流程"
        }
        
    except Exception as e:
        logger.error(f"启动CodeBuddy认证失败: {e}")
        return {
            "success": False,
            "error": "auth_start_failed", 
            "message": f"认证启动失败: {str(e)}"
        }

async def poll_codebuddy_auth_status(auth_state: str) -> Dict[str, Any]:
    """轮询CodeBuddy认证状态"""
    try:
        headers = get_codebuddy_headers()
        url = f"{CODEBUDDY_AUTH_TOKEN_ENDPOINT}?state={auth_state}"
        
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('code') == 11217:
                    # 仍在等待登录
                    return {
                        "status": "pending",
                        "message": result.get('msg', 'login ing...'),
                        "code": result.get('code')
                    }
                elif result.get('code') == 0 and result.get('data') and result.get('data', {}).get('accessToken'):
                    # 认证成功，获得token
                    data = result.get('data', {})
                    return {
                        "status": "success",
                        "message": "认证成功！",
                        "token_data": {
                            "access_token": data.get('accessToken'),
                            "bearer_token": data.get('accessToken'),
                            "token_type": data.get('tokenType', 'Bearer'),
                            "expires_in": data.get('expiresIn'),
                            "refresh_token": data.get('refreshToken'),
                            "session_state": data.get('sessionState'),
                            "scope": data.get('scope'),
                            "domain": data.get('domain'),
                            "full_response": result
                        }
                    }
                else:
                    # 其他状态码
                    return {
                        "status": "unknown",
                        "message": result.get('msg', 'Unknown status'),
                        "code": result.get('code'),
                        "response": result
                    }
            else:
                return {
                    "status": "error",
                    "message": f"API请求失败，状态码: {response.status_code}",
                    "response_text": response.text
                }
                
    except Exception as e:
        logger.error(f"轮询认证状态失败: {e}")
        return {
            "status": "error",
            "message": f"轮询失败: {str(e)}"
        }

async def save_codebuddy_token(token_data: Dict[str, Any]) -> bool:
    """保存CodeBuddy token到文件"""
    try:
        from codebuddy_token_manager import codebuddy_token_manager
        
        # 添加创建时间
        token_data["created_at"] = int(time.time())
        
        # 从JWT中解析用户信息
        bearer_token = token_data.get("access_token") or token_data.get("bearer_token")
        user_id = "unknown"
        
        try:
            if bearer_token:
                payload_part = bearer_token.split('.')[1]
                payload_part += '=' * (4 - len(payload_part) % 4)
                payload = base64.urlsafe_b64decode(payload_part)
                jwt_data = json.loads(payload)
                user_id = jwt_data.get('sub') or jwt_data.get('preferred_username') or jwt_data.get('email') or "unknown"
                logger.info(f"从JWT中解析出用户ID: {user_id}")
        except Exception as e:
            logger.warning(f"无法从JWT中解析用户信息: {e}")
            user_id = token_data.get('domain', 'unknown')
        
        # 使用token管理器保存
        success = codebuddy_token_manager.add_credential(
            bearer_token=bearer_token,
            user_id=user_id,
            filename=f"codebuddy_oauth_{int(time.time())}.json"
        )
        
        if success:
            logger.info(f"成功保存CodeBuddy token，用户: {user_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"保存CodeBuddy token失败: {e}")
        return False

# --- API Endpoints ---
@router.get("/auth/start", summary="Start CodeBuddy Authentication")
async def start_device_auth():
    """启动CodeBuddy认证流程"""
    try:
        logger.info("开始启动CodeBuddy认证流程...")
        
        # 尝试真实的CodeBuddy认证API
        real_auth_result = await start_codebuddy_auth()
        
        if real_auth_result.get('success'):
            logger.info("真实CodeBuddy认证API启动成功!")
            return real_auth_result
        else:
            logger.warning(f"真实认证API失败: {real_auth_result}")
            return real_auth_result
        
    except Exception as e:
        logger.error(f"认证启动过程发生异常: {e}")
        return {
            "success": False,
            "error": "Unexpected error",
            "message": f"认证启动失败: {str(e)}"
        }

@router.post("/auth/poll", summary="Poll for OAuth token")
async def poll_for_token(
    device_code: str = Body(None, embed=True),
    code_verifier: str = Body(None, embed=True),
    auth_state: str = Body(None, embed=True)
):
    """轮询CodeBuddy token端点"""
    from codebuddy_token_manager import codebuddy_token_manager
    
    # 如果有auth_state，说明是真实的CodeBuddy认证流程
    if auth_state:
        logger.info(f"轮询真实CodeBuddy认证状态: {auth_state}")
        poll_result = await poll_codebuddy_auth_status(auth_state)
        
        if poll_result.get('status') == 'success':
            # 认证成功，保存token
            token_data = poll_result.get('token_data', {})
            if token_data:
                # 提取token信息
                bearer_token = token_data.get('access_token') or token_data.get('bearer_token')
                if bearer_token:
                    # 保存token
                    token_saved = await save_codebuddy_token(token_data)
                    return JSONResponse(content={
                        "access_token": bearer_token,
                        "token_type": token_data.get('token_type', 'Bearer'),
                        "expires_in": token_data.get('expires_in'),
                        "refresh_token": token_data.get('refresh_token'),
                        "scope": token_data.get('scope'),
                        "saved": token_saved,
                        "message": "认证成功！🎉",
                        "user_info": token_data,
                        "domain": token_data.get('domain')
                    }, status_code=200)
                else:
                    return JSONResponse(content={
                        "error": "invalid_token_response",
                        "error_description": "API返回的响应中没有找到token"
                    }, status_code=400)
        elif poll_result.get('status') == 'pending':
            # 仍在等待
            return JSONResponse(content={
                "error": "authorization_pending",
                "error_description": poll_result.get('message', '等待用户登录...'),
                "code": poll_result.get('code')
            }, status_code=400)
        else:
            # 错误状态
            return JSONResponse(content={
                "error": "auth_error",
                "error_description": poll_result.get('message', '认证过程发生错误'),
                "details": poll_result
            }, status_code=400)
    else:
        return JSONResponse(content={
            "error": "missing_parameters",
            "error_description": "缺少必要的参数：auth_state"
        }, status_code=400)

@router.get("/auth/callback", summary="OAuth2 callback endpoint")
async def oauth_callback(code: str = None, state: str = None, error: str = None):
    """OAuth2回调端点"""
    if error:
        return JSONResponse(
            content={"error": error, "error_description": "授权被拒绝或出现错误"},
            status_code=400
        )
    
    return JSONResponse(
        content={
            "message": "授权成功！请返回应用程序。",
            "code": code,
            "state": state
        }
    )