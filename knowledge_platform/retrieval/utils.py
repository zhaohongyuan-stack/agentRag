"""
工具函数 — ModelScope 模型下载等。
"""


def modelscope_download(model_name: str) -> str:
    """从 ModelScope 下载模型并返回本地路径，已缓存则直接返回"""
    from modelscope import snapshot_download
    print(f"  [ModelScope] {model_name} ...")
    return snapshot_download(model_name)
