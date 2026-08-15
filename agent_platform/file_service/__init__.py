"""
文件服务 — 真实业务文件浏览/预览/下载

提供制度文库所需的真实文件目录树、HTML 预览、原文件下载能力。
"""

from .file_server import router, _register_router

__all__ = ["router", "_register_router"]
