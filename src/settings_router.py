"""
Settings Router - For loading and saving .env configurations
"""
import os
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any

from .auth import authenticate
from config import get_active_config, update_settings
from .usage_stats_manager import usage_stats_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# 中文标签映射
SETTING_LABELS = {
    "CODEBUDDY_HOST": "服务主机地址",
    "CODEBUDDY_PORT": "服务端口",
    "CODEBUDDY_PASSWORD": "API 服务访问密码",
    "CODEBUDDY_API_ENDPOINT": "CodeBuddy 官方API端点",
    "CODEBUDDY_CREDS_DIR": "凭证文件目录",
    "CODEBUDDY_LOG_LEVEL": "日志级别",
    "CODEBUDDY_MODELS": "可用模型列表 (逗号分隔)",
    "CODEBUDDY_ROTATION_COUNT": "凭证轮换频率 (N次请求/凭证)"
}

class Settings(BaseModel):
    settings: Dict[str, Any]

@router.get("/settings", summary="Get all current active settings and labels")
async def get_settings(_token: str = Depends(authenticate)):
    """Returns the current config and their Chinese labels."""
    try:
        return {
            "settings": get_active_config(),
            "labels": SETTING_LABELS
        }
    except Exception as e:
        logger.error(f"Error retrieving active config: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve settings.")

@router.post("/settings", summary="Save and hot-reload settings")
async def save_settings(new_settings: Settings, _token: str = Depends(authenticate)):
    """Saves settings to config.json and hot-reloads them into memory."""
    try:
        update_settings(new_settings.settings)
        return {"message": "设置已保存并成功热加载！"}
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        raise HTTPException(status_code=500, detail="无法保存设置文件。")

@router.get("/stats", summary="Get usage statistics")
async def get_usage_stats(_token: str = Depends(authenticate)):
    """Returns usage statistics for models and credentials."""
    try:
        return usage_stats_manager.get_stats()
    except Exception as e:
        logger.error(f"Error retrieving usage stats: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve usage statistics.")
