"""
Serves the frontend for CodeBuddy2API management interface.
"""
from fastapi import APIRouter
from fastapi.responses import FileResponse
import os

router = APIRouter()

# Get the absolute path to the admin interface file
HTML_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend", "admin.html")

@router.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_frontend():
    """Serves the CodeBuddy2API admin interface."""
    if not os.path.exists(HTML_FILE_PATH):
        return "Frontend file not found. Please ensure frontend/admin.html exists."
    return FileResponse(HTML_FILE_PATH, media_type="text/html", headers={"Content-Type": "text/html; charset=utf-8"})

@router.get("/admin", response_class=FileResponse, include_in_schema=False) 
async def serve_admin():
    """Alternative route for admin interface."""
    return await serve_frontend()