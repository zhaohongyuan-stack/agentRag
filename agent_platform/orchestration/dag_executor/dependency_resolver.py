"""
DAG 执行器 — 依赖关系解析与拓扑排序

负责分析子任务间的依赖关系，提供拓扑排序（分层）、就绪任务筛选、
循环依赖检测与后继任务查询等能力，是 DagExecutor 调度决策的基础。

设计要点:
  1. 分层拓扑: topological_sort 返回 List[List[task_id]]，同一层内任务
     互不依赖，可并行执行，最大化并行度
  2. 环检测前置: 执行前必须调用 has_cycle / topological_sort 检测循环依赖，
     发现环时抛出 ValueError 避免死循环
  3. 鲁棒性: 引用不存在的依赖 task_id 时记录告警并忽略，不阻断执行
  4. 无副作用: 所有方法均为纯查询，不修改传入的 DagTask 状态

使用方式:
    from agent_platform.orchestration.dag_executor import DependencyResolver

    resolver = DependencyResolver()
    layers = resolver.topological_sort(tasks)  # [["t1"], ["t2", "t3"], ["t4"]]
"""

import logging
from typing import Dict, List, Set

from .task_models import DagTask, TaskStatus

logger = logging.getLogger(__name__)


class DependencyResolver:
    """
    依赖关系解析器 — 分析 DAG 任务间的依赖结构

    提供拓扑排序、就绪任务筛选、环检测与后继查询，供 DagExecutor
    在调度时确定执行顺序与可并行任务集合。
    """

    # ============================================================
    # 内部辅助
    # ============================================================

    @staticmethod
    def _build_task_index(tasks: List[DagTask]) -> Dict[str, DagTask]:
        """构建 task_id -> DagTask 的索引，便于快速查找"""
        index: Dict[str, DagTask] = {}
        for task in tasks:
            if task.task_id in index:
                logger.warning("发现重复 task_id=%s，后者将覆盖前者", task.task_id)
            index[task.task_id] = task
        return index

    def _normalize_dependencies(
        self, tasks: List[DagTask], index: Dict[str, DagTask]
    ) -> Dict[str, Set[str]]:
        """
        规范化依赖关系，过滤掉引用不存在任务的依赖

        Returns:
            task_id -> 该任务依赖的有效 task_id 集合
        """
        deps_map: Dict[str, Set[str]] = {}
        for task in tasks:
            valid_deps: Set[str] = set()
            for dep_id in task.dependencies:
                if dep_id in index:
                    if dep_id == task.task_id:
                        logger.warning(
                            "任务 task_id=%s 依赖自身，已忽略该自环",
                            task.task_id,
                        )
                        continue
                    valid_deps.add(dep_id)
                else:
                    logger.warning(
                        "任务 task_id=%s 依赖不存在的任务 dep_id=%s，已忽略",
                        task.task_id,
                        dep_id,
                    )
            deps_map[task.task_id] = valid_deps
        return deps_map

    # ============================================================
    # 拓扑排序
    # ============================================================

    def topological_sort(self, tasks: List[DagTask]) -> List[List[str]]:
        """
        分层拓扑排序，返回可并行执行的层级计划

        采用 Kahn 算法的分层变体：每轮取出当前入度为 0 的所有任务构成一层，
        移除后更新后继入度，直至全部处理完毕。同一层内的任务互不依赖，
        可安全并行执行。

        Args:
            tasks: 待排序的任务列表

        Returns:
            分层执行计划 List[List[task_id]]，外层顺序即执行顺序，
            内层任务可并行

        Raises:
            ValueError: 检测到循环依赖时抛出
        """
        if not tasks:
            return []

        index = self._build_task_index(tasks)
        deps_map = self._normalize_dependencies(tasks, index)

        # 计算入度（依赖的任务数）
        in_degree: Dict[str, int] = {
            tid: len(deps_map[tid]) for tid in index
        }
        # 构建后继列表：被依赖者 -> 依赖它的任务集合
        successors: Dict[str, List[str]] = {tid: [] for tid in index}
        for tid, deps in deps_map.items():
            for dep_id in deps:
                successors[dep_id].append(tid)

        layers: List[List[str]] = []
        processed: Set[str] = set()

        # 按层处理，每层取当前入度为 0 的任务
        while len(processed) < len(index):
            # 当前层：入度为 0 且尚未处理；按 task_id 排序保证输出稳定
            current_layer = sorted(
                tid
                for tid in index
                if tid not in processed and in_degree[tid] == 0
            )
            if not current_layer:
                # 剩余任务均存在依赖未满足 -> 存在环
                remaining = sorted(set(index) - processed)
                raise ValueError(
                    f"检测到循环依赖，无法完成拓扑排序，"
                    f"涉及任务: {remaining}"
                )
            layers.append(current_layer)
            for tid in current_layer:
                processed.add(tid)
                # 移除该任务后，后继入度减一
                for succ in successors[tid]:
                    in_degree[succ] -= 1

        logger.debug("拓扑排序完成，共 %d 层", len(layers))
        return layers

    # ============================================================
    # 就绪任务
    # ============================================================

    def get_ready_tasks(self, tasks: List[DagTask]) -> List[str]:
        """
        获取所有可执行的任务（PENDING 且依赖已全部 COMPLETED）

        Args:
            tasks: 任务列表

        Returns:
            可执行任务的 task_id 列表
        """
        index = self._build_task_index(tasks)
        ready: List[str] = []
        for task in tasks:
            if task.status != TaskStatus.PENDING:
                continue
            # 所有依赖均存在且已完成
            if all(
                dep_id in index and index[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            ):
                ready.append(task.task_id)
        return ready

    # ============================================================
    # 环检测
    # ============================================================

    def has_cycle(self, tasks: List[DagTask]) -> bool:
        """
        检测任务依赖图中是否存在循环依赖

        采用 DFS 三色标记法（白/灰/黑）：
          - 白色：尚未访问
          - 灰色：正在当前 DFS 路径中访问
          - 黑色：已完成访问
        若在搜索过程中遇到灰色节点，说明存在回边即循环依赖。

        Args:
            tasks: 任务列表

        Returns:
            True 表示存在循环依赖，False 表示无环
        """
        if not tasks:
            return False

        index = self._build_task_index(tasks)
        deps_map = self._normalize_dependencies(tasks, index)

        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {tid: WHITE for tid in index}

        def _dfs(node: str) -> bool:
            """从 node 出发深度优先搜索，返回是否发现环"""
            color[node] = GRAY
            for dep_id in deps_map[node]:
                if color[dep_id] == GRAY:
                    # 回边 -> 环
                    return True
                if color[dep_id] == WHITE and _dfs(dep_id):
                    return True
            color[node] = BLACK
            return False

        for tid in index:
            if color[tid] == WHITE:
                if _dfs(tid):
                    logger.warning("检测到循环依赖，起始任务 task_id=%s", tid)
                    return True
        return False

    # ============================================================
    # 后继查询
    # ============================================================

    def get_dependents(
        self, task_id: str, tasks: List[DagTask]
    ) -> List[str]:
        """
        获取依赖指定任务的后继任务

        即 dependencies 列表中包含 task_id 的所有任务。

        Args:
            task_id: 被依赖的任务 ID
            tasks: 任务列表

        Returns:
            依赖该任务的后继 task_id 列表
        """
        dependents: List[str] = []
        for task in tasks:
            if task.task_id == task_id:
                continue
            if task_id in task.dependencies:
                dependents.append(task.task_id)
        return dependents
