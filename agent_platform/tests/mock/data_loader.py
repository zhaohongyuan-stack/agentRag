"""
Mock 数据加载器 — 从 contracts/examples/ 加载样例数据

职责：
  - 从 contracts/examples/retrieval_requests/ 加载检索请求样例（JSON）
  - 从 contracts/examples/retrieval_responses/ 加载检索响应样例（JSON 数组）
  - 提供按 scenario 名称查找请求和响应的方法

设计要点：
  - 样例数据可能尚未创建完毕，加载器需优雅处理文件不存在的情况
  - 每个请求文件可选包含 "scenario" 字段用于场景标识
  - 响应文件按文件名与请求文件配对（同名文件视为一对）
  - 文件名前缀也作为场景标识的回退依据（如 timeout.json -> scenario "timeout"）

用法：
    from agent_platform.tests.mock.data_loader import DataLoader

    loader = DataLoader()
    requests = loader.load_requests()      # {文件名: 请求dict}
    responses = loader.load_responses()    # {文件名: 响应list}
    req, resp = loader.get_scenario("normal")  # (请求dict, 响应list)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class DataLoader:
    """
    Mock 样例数据加载器

    从 contracts/examples/ 目录加载 retrieval_requests/ 和 retrieval_responses/
    下的 JSON 文件，支持按 scenario 名称查找配对的请求与响应。

    Attributes:
        examples_dir: contracts/examples/ 根目录路径
    """

    def __init__(self, examples_dir: Optional[str] = None):
        """
        初始化数据加载器

        Args:
            examples_dir: contracts/examples/ 目录路径
                          为 None 时自动推导为项目根目录下的 contracts/examples/
        """
        if examples_dir is not None:
            self.examples_dir = Path(examples_dir)
        else:
            # 自动推导项目根目录：data_loader.py 位于 agent_platform/tests/mock/
            # 向上回溯 4 层即为项目根目录 ragagent/
            project_root = Path(__file__).resolve().parents[3]
            self.examples_dir = project_root / "contracts" / "examples"

        self._requests_dir = self.examples_dir / "retrieval_requests"
        self._responses_dir = self.examples_dir / "retrieval_responses"

        # 缓存
        self._requests_cache: Optional[Dict[str, dict]] = None
        self._responses_cache: Optional[Dict[str, list]] = None

    # ============================================================
    # 加载请求
    # ============================================================
    def load_requests(self) -> Dict[str, dict]:
        """
        加载所有检索请求样例

        遍历 retrieval_requests/ 目录下的所有 .json 文件，
        每个文件解析为一个请求 dict。

        Returns:
            {文件名（不含扩展名）: 请求dict}
            如果目录不存在或没有文件，返回空 dict
        """
        if self._requests_cache is not None:
            return self._requests_cache

        result: Dict[str, dict] = {}

        if not self._requests_dir.exists():
            return result

        for fpath in sorted(self._requests_dir.glob("*.json")):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    # 文件名（不含扩展名）作为 key
                    result[fpath.stem] = data
            except (json.JSONDecodeError, OSError) as e:
                # 优雅处理：跳过无法解析的文件
                print(f"  [DataLoader] 跳过请求文件 {fpath.name}: {e}")

        self._requests_cache = result
        return result

    # ============================================================
    # 加载响应
    # ============================================================
    def load_responses(self) -> Dict[str, list]:
        """
        加载所有检索响应样例

        遍历 retrieval_responses/ 目录下的所有 .json 文件，
        每个文件解析为 RetrievalHit 列表（JSON 数组）。

        Returns:
            {文件名（不含扩展名）: 响应list}
            如果目录不存在或没有文件，返回空 dict
        """
        if self._responses_cache is not None:
            return self._responses_cache

        result: Dict[str, list] = {}

        if not self._responses_dir.exists():
            return result

        for fpath in sorted(self._responses_dir.glob("*.json")):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 响应应为 JSON 数组；单个对象也包装为列表
                if isinstance(data, list):
                    result[fpath.stem] = data
                elif isinstance(data, dict):
                    result[fpath.stem] = [data]
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [DataLoader] 跳过响应文件 {fpath.name}: {e}")

        self._responses_cache = result
        return result

    # ============================================================
    # 按场景查找
    # ============================================================
    def get_scenario(
        self, scenario_name: str
    ) -> Tuple[Optional[dict], Optional[list]]:
        """
        按 scenario 名称查找配对的请求与响应

        查找逻辑：
        1. 在请求中查找 scenario 字段匹配 scenario_name 的文件
        2. 如果找不到，尝试用文件名前缀匹配（如 timeout.json -> "timeout"）
        3. 响应按同名文件配对（请求文件名为 X，则查找响应文件名也为 X）
        4. 若无同名响应文件，尝试查找任意一个同 scenario 的响应

        Args:
            scenario_name: 场景名称，如 "normal", "empty", "timeout",
                           "version_conflict", "partial_failure"

        Returns:
            (请求dict, 响应list)
            如果找不到请求，返回 (None, None)
            如果找到请求但无配对响应，返回 (请求dict, None)
        """
        requests = self.load_requests()
        responses = self.load_responses()

        # 1. 通过 scenario 字段匹配
        matched_request_key = None
        for key, req in requests.items():
            if req.get("scenario") == scenario_name:
                matched_request_key = key
                break

        # 2. 回退：通过文件名前缀匹配
        if matched_request_key is None:
            for key in requests:
                # 文件名以 scenario_name 开头，或完全等于 scenario_name
                if key == scenario_name or key.startswith(scenario_name + "_"):
                    matched_request_key = key
                    break

        if matched_request_key is None:
            return (None, None)

        request = requests[matched_request_key]

        # 3. 同名文件配对查找响应
        response = responses.get(matched_request_key)

        # 4. 回退：查找任意 scenario 字段匹配的响应（响应文件不包含 scenario，
        #    只能通过文件名匹配，已在步骤 3 完成）

        return (request, response)

    # ============================================================
    # 便利方法
    # ============================================================
    def list_scenarios(self) -> List[str]:
        """
        列出所有可用的 scenario 名称（从请求文件的 scenario 字段提取）

        Returns:
            场景名称列表，去重后排序
        """
        requests = self.load_requests()
        scenarios = set()
        for req in requests.values():
            scenario = req.get("scenario")
            if scenario:
                scenarios.add(scenario)
        return sorted(scenarios)

    def reload(self) -> "DataLoader":
        """清除缓存，强制重新加载（链式调用）"""
        self._requests_cache = None
        self._responses_cache = None
        return self

    def __repr__(self) -> str:
        req_count = len(self._requests_cache) if self._requests_cache else "?"
        resp_count = len(self._responses_cache) if self._responses_cache else "?"
        return (
            f"DataLoader(examples_dir={self.examples_dir}, "
            f"requests={req_count}, responses={resp_count})"
        )
