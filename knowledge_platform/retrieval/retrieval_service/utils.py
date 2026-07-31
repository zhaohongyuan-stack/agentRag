"""
工具函数 — ModelScope 模型下载等。

优化：优先使用本地缓存，避免每次启动都联网验证。
"""

import os
from pathlib import Path
from typing import Optional


def _get_local_cache_path(model_name: str) -> Optional[str]:
    """
    检查 ModelScope 本地缓存，如果模型已存在则返回路径。

    ModelScope 缓存路径格式:
        ~/.cache/modelscope/models/{org}--{model}/snapshots/{revision}/
    """
    cache_base = Path.home() / ".cache" / "modelscope" / "models"
    # BAAI/bge-small-zh-v1.5 → BAAI--bge-small-zh-v1.5
    model_dir_name = model_name.replace("/", "--")
    model_dir = cache_base / model_dir_name / "snapshots"

    if not model_dir.exists():
        return None

    # 取第一个 snapshot（通常是 master）
    snapshots = list(model_dir.iterdir())
    if not snapshots:
        return None

    # 验证目录中有模型文件
    snapshot_path = snapshots[0]
    has_model = any(
        (snapshot_path / f).exists()
        for f in ["model.safetensors", "pytorch_model.bin"]
    )
    if has_model:
        return str(snapshot_path)

    return None


def modelscope_download(model_name: str) -> str:
    """
    从 ModelScope 下载模型并返回本地路径。

    优化流程:
    1. 先检查本地缓存，命中则直接返回（不联网）
    2. 缓存未命中则尝试下载（尊重用户代理设置）
    3. 下载失败时给出清晰错误提示
    """
    # ── 步骤1: 检查本地缓存 ──
    local_path = _get_local_cache_path(model_name)
    if local_path:
        print(f"  [ModelScope] 本地缓存命中: {model_name}")
        return local_path

    # ── 步骤2: 尝试下载 ──
    print(f"  [ModelScope] 本地缓存未命中，尝试下载: {model_name} ...")
    try:
        from modelscope import snapshot_download

        # 检查代理状态（用于日志提示，不修改环境变量）
        proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY", "")
        if proxy:
            print(f"  [ModelScope] 使用代理: {proxy}")

        result = snapshot_download(model_name)
        print(f"  [ModelScope] 下载完成: {result}")
        return result
    except ImportError:
        print(f"  [ModelScope] modelscope 未安装，尝试 HuggingFace 缓存 ...")
        # 尝试 HuggingFace 缓存路径
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        hf_dir_name = f"models--{model_name.replace('/', '--')}"
        hf_model_dir = hf_cache / hf_dir_name / "snapshots"
        if hf_model_dir.exists():
            snapshots = list(hf_model_dir.iterdir())
            if snapshots:
                return str(snapshots[0])
        raise RuntimeError(
            f"模型 {model_name} 未找到。请先安装 modelscope: pip install modelscope"
        )
    except Exception as e:
        # 下载失败，再次检查本地缓存（可能其他进程刚下载完）
        local_path = _get_local_cache_path(model_name)
        if local_path:
            print(f"  [ModelScope] 下载失败但本地缓存可用: {model_name}")
            return local_path
        raise RuntimeError(f"模型 {model_name} 下载失败且本地无缓存: {e}")
