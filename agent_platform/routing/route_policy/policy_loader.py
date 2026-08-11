"""
策略加载器 — 从 YAML 加载执行路径配置

支持:
  - 从 execution_paths.yaml 加载 P0-P4 执行路径
  - 热更新（reload 重新从文件加载）
  - 文件不存在时回退到内置默认路径

与 route_policy.py 中的 DEFAULT_EXECUTION_PATHS 对齐。
"""

import os
from typing import Dict, Optional

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from .path_table import DEFAULT_EXECUTION_PATHS, ExecutionPath


# 默认 YAML 配置文件路径（与本文件同目录）
_DEFAULT_CONFIG_FILENAME = "execution_paths.yaml"


class PolicyLoader:
    """
    执行路径策略加载器

    从 YAML 文件加载 P0-P4 执行路径配置。
    文件不存在时使用内置默认路径。

    用法:
        loader = PolicyLoader()
        paths = loader.load()           # 首次加载
        paths = loader.reload()         # 热更新
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: YAML 配置文件路径。
                         为 None 时自动查找同目录下的 execution_paths.yaml。
        """
        if config_path:
            self._config_path = config_path
        else:
            # 默认查找同目录下的 execution_paths.yaml
            module_dir = os.path.dirname(os.path.abspath(__file__))
            self._config_path = os.path.join(module_dir, _DEFAULT_CONFIG_FILENAME)

        self._paths: Dict[str, ExecutionPath] = {}

    def load(self, path: Optional[str] = None) -> Dict[str, ExecutionPath]:
        """
        加载执行路径配置

        Args:
            path: 指定 YAML 文件路径，为 None 时使用初始化时设定的路径

        Returns:
            执行路径字典 {path_id: ExecutionPath}
        """
        if path:
            self._config_path = path

        self._paths = self._load_from_file(self._config_path)
        return self._paths

    def reload(self) -> Dict[str, ExecutionPath]:
        """
        热更新: 重新从文件加载执行路径配置

        Returns:
            重新加载后的执行路径字典
        """
        return self.load()

    def get(self, path_id: str) -> Optional[ExecutionPath]:
        """
        获取指定执行路径

        Args:
            path_id: 执行路径 ID（P0-P4）

        Returns:
            ExecutionPath 对象，未找到返回 None
        """
        return self._paths.get(path_id)

    def list_path_ids(self) -> list:
        """列出所有执行路径 ID"""
        return sorted(self._paths.keys())

    # ----------------------------------------------------------
    # 内部方法: 从文件加载
    # ----------------------------------------------------------
    def _load_from_file(self, path: str) -> Dict[str, ExecutionPath]:
        """
        从 YAML 文件加载执行路径

        文件不存在或解析失败时回退到内置默认路径。

        Args:
            path: YAML 文件路径

        Returns:
            执行路径字典
        """
        # 文件不存在 → 使用内置默认路径
        if not path or not os.path.exists(path):
            return dict(DEFAULT_EXECUTION_PATHS)

        # 无 YAML 库 → 使用内置默认路径
        if not _HAS_YAML:
            return dict(DEFAULT_EXECUTION_PATHS)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            # 解析失败 → 回退到默认路径
            return dict(DEFAULT_EXECUTION_PATHS)

        if not data or "execution_paths" not in data:
            return dict(DEFAULT_EXECUTION_PATHS)

        # 解析 YAML 数据为 ExecutionPath 对象
        paths: Dict[str, ExecutionPath] = {}
        for path_id, cfg in data["execution_paths"].items():
            if not isinstance(cfg, dict):
                continue
            paths[path_id] = ExecutionPath(
                path_id=path_id,
                description=cfg.get("description", ""),
                channels=cfg.get("channels", []),
                top_k=cfg.get("top_k", 10),
                rerank=cfg.get("rerank", False),
                rerank_top_n=cfg.get("rerank_top_n", 0),
                budget_ms=cfg.get("budget_ms", 5000),
                max_retries=cfg.get("max_retries", 1),
                need_decomposition=cfg.get("need_decomposition", False),
                cache_first=cfg.get("cache_first", False),
                retrieval=cfg.get("retrieval", True),
            )

        # 确保至少有 P0-P4 的默认路径（合并: YAML 覆盖默认值）
        for default_id, default_path in DEFAULT_EXECUTION_PATHS.items():
            if default_id not in paths:
                paths[default_id] = default_path

        return paths
