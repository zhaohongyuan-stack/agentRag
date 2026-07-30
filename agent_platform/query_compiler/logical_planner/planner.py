"""
逻辑计划生成 — 将 QueryIR 转换为逻辑执行计划

职责:
  1. 将检索算子编排为执行阶段（stages）
  2. 根据声明依赖关系决定并行/串行
  3. 为每个阶段标注完成条件

逻辑计划不涉及具体通道参数（top_k / timeout），仅描述执行结构与依赖。

模式参考: routing/route_policy/ 的 dataclass + to_dict 风格
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import List

from ..query_ir.ir_builder import Dependency, Operator, QueryIR


logger = logging.getLogger(__name__)


@dataclass
class PlanStage:
    """
    逻辑计划阶段

    一个阶段包含一组可并行执行的检索算子，
    以及该阶段依赖的前序阶段名称。

    Attributes:
        name: 阶段名称（如 retrieve / verify_xxx）
        operators: 该阶段的检索算子列表
        dependencies: 依赖的前序阶段名称列表
        can_parallel: 阶段内算子是否可并行
        completion_condition: 阶段完成条件描述
    """

    name: str
    operators: List[Operator] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    can_parallel: bool = True
    completion_condition: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "operators": [o.to_dict() for o in self.operators],
            "dependencies": list(self.dependencies),
            "can_parallel": self.can_parallel,
            "completion_condition": self.completion_condition,
        }


@dataclass
class LogicalPlan:
    """
    逻辑执行计划

    由 LogicalPlanner 生成，包含若干执行阶段，
    描述算子间的编排结构与依赖关系。
    """

    plan_id: str
    intent: str
    stages: List[PlanStage] = field(default_factory=list)
    risk_level: str = "medium"

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "intent": self.intent,
            "stages": [s.to_dict() for s in self.stages],
            "risk_level": self.risk_level,
        }


class LogicalPlanner:
    """
    逻辑计划生成器

    将 QueryIR 编排为逻辑执行计划。

    编排策略:
      - 无依赖或并行依赖 → 单阶段并行执行所有算子
      - 存在串行依赖（高风险多声明） → 检索阶段 + 逐声明校验阶段

    用法:
        planner = LogicalPlanner()
        plan = planner.plan(query_ir)
        print(plan.stages)
    """

    def plan(self, query_ir: QueryIR) -> LogicalPlan:
        """
        生成逻辑计划

        Args:
            query_ir: 查询中间表示

        Returns:
            LogicalPlan 对象
        """
        plan_id = f"lp-{uuid.uuid4().hex[:8]}"
        operators = query_ir.retrieval_operators
        dependencies = query_ir.dependencies

        if not operators:
            logger.warning("QueryIR 无检索算子，生成空逻辑计划")

        # 判断是否存在串行依赖（高风险多声明场景）
        has_sequential = any(d.type == "sequential" for d in dependencies)

        if has_sequential and operators:
            stages = self._build_sequential_stages(operators, dependencies)
        else:
            stages = self._build_parallel_stage(operators)

        plan = LogicalPlan(
            plan_id=plan_id,
            intent=query_ir.intent,
            stages=stages,
            risk_level=query_ir.risk_level,
        )
        logger.info(
            "逻辑计划生成完成: plan_id=%s, intent=%s, stages=%d",
            plan_id,
            query_ir.intent,
            len(stages),
        )
        return plan

    # ============================================================
    # 内部方法
    # ============================================================

    def _build_parallel_stage(self, operators: List[Operator]) -> List[PlanStage]:
        """
        构建单阶段并行计划

        所有算子放在一个 "retrieve" 阶段中并行执行。
        """
        stage = PlanStage(
            name="retrieve",
            operators=list(operators),
            dependencies=[],
            can_parallel=True,
            completion_condition="all_operators_returned",
        )
        return [stage]

    def _build_sequential_stages(
        self,
        operators: List[Operator],
        dependencies: List[Dependency],
    ) -> List[PlanStage]:
        """
        按串行依赖构建多阶段计划

        策略:
          1. 阶段1 "retrieve": 执行全部检索算子（可并行）
          2. 后续阶段: 按依赖链逐声明校验（串行），每个校验阶段依赖前一个
        """
        stages: List[PlanStage] = []

        # 阶段1：检索（所有算子并行）
        stages.append(
            PlanStage(
                name="retrieve",
                operators=list(operators),
                dependencies=[],
                can_parallel=True,
                completion_condition="all_operators_returned",
            )
        )

        # 后续阶段：逐声明校验（串行）
        prev_name = "retrieve"
        for dep in dependencies:
            stage_name = f"verify_{dep.to_claim}"
            stages.append(
                PlanStage(
                    name=stage_name,
                    operators=[],  # 校验阶段无检索算子
                    dependencies=[prev_name],
                    can_parallel=False,
                    completion_condition=f"claim_{dep.to_claim}_resolved",
                )
            )
            prev_name = stage_name

        return stages
