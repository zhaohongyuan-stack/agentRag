"""
冲突检测器 — 检测证据项之间的各类冲突

职责:
  1. 数值不一致检测（NUMERIC_MISMATCH）: 同一指标在不同证据中出现不同数值
  2. 版本冲突检测（VERSION_CONFLICT）: 同一文档存在 active 与 superseded 等不同版本状态
  3. 适用范围重叠检测（SCOPE_OVERLAP）: 两个不同规定的适用范围相互重叠
  4. 效力冲突检测（AUTHORITY_CONFLICT）: 不同效力层级文件对同一问题有不同规定
  5. 时效冲突检测（TEMPORAL_CONFLICT）: 同一规定存在多个生效日期版本

设计要点:
  - 复用 evidence_assembler/builder.py 中的 EvidenceItem
  - 数值提取使用正则表达式（匹配百分比 8%、8.5% 及普通数值）
  - 不自动解决冲突，仅检测并标记，解决建议由 ConflictResolver 提供
  - 各检测方法相互独立，由 detect() 汇总全部结果
  - 所有日志使用 logging.getLogger(__name__)
"""

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from agent_platform.evidence.evidence_assembler.builder import EvidenceItem

from .conflict_types import (
    CONFLICT_PRIORITY,
    Conflict,
    ConflictType,
)

logger = logging.getLogger(__name__)


class ConflictDetector:
    """
    证据冲突检测器

    对一组 EvidenceItem 执行五类冲突检测，返回 Conflict 列表。
    各检测方法聚焦不同维度，互不依赖:

      - NUMERIC_MISMATCH: 关注同一指标的数值差异
      - VERSION_CONFLICT: 关注同一文档的版本状态混用
      - SCOPE_OVERLAP: 关注不同规定的适用范围交叉
      - AUTHORITY_CONFLICT: 关注不同效力层级的规定差异
      - TEMPORAL_CONFLICT: 关注同一规定的生效日期版本
    """

    # 数值提取正则: 匹配百分比（8%、8.5%、8％），捕获数值部分
    _PERCENT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*[%％]")

    # 数值提取正则: 匹配普通数值（前后不能紧跟小数点或字母数字，避免拆分小数/年份片段）
    _NUMBER_PATTERN = re.compile(r"(?<![.\w])(\d+(?:\.\d+)?)(?![.\w])")

    # 指标上下文关键词（出现在数值之前，用于切分出指标名称）
    _METRIC_KEYWORDS: List[str] = [
        "不得低于", "不低于", "不得高于", "不高于",
        "不得超过", "不超过", "应不低于", "应不高于",
        "应当不低于", "应当不高于", "至少", "最高",
        "最低", "应达到", "达到", "为", "等于",
        "不大于", "不小于", "大于", "小于",
    ]

    # 句段分隔符（用于从上下文中提取指标短语的最后一段）
    # 包含中文标点、英文标点、空白、省略号（… U+2026）
    _PHRASE_SEPARATORS = r"[，。、；;：:！？\n\r\t（）()【】\[\] …]+"

    # ============================================================
    # 公共方法
    # ============================================================

    def detect(self, evidence_items: List[EvidenceItem]) -> List[Conflict]:
        """
        检测所有冲突类型

        依次执行五类检测，汇总结果。各检测方法独立运行，互不影响。

        Args:
            evidence_items: EvidenceItem 列表

        Returns:
            检测到的 Conflict 列表（可能为空）
        """
        if not evidence_items:
            return []

        conflicts: List[Conflict] = []

        conflicts.extend(self._detect_numeric_mismatch(evidence_items))
        conflicts.extend(self._detect_version_conflict(evidence_items))
        conflicts.extend(self._detect_scope_overlap(evidence_items))
        conflicts.extend(self._detect_authority_conflict(evidence_items))
        conflicts.extend(self._detect_temporal_conflict(evidence_items))

        # 去重: 同类型且涉及相同证据集合的冲突只保留一条
        conflicts = self._deduplicate_conflicts(conflicts)

        logger.debug(
            "冲突检测完成: 共 %d 条证据，检出 %d 条冲突",
            len(evidence_items),
            len(conflicts),
        )
        return conflicts

    # ============================================================
    # 内部方法 — 各类冲突检测
    # ============================================================

    def _detect_numeric_mismatch(
        self, evidence_items: List[EvidenceItem]
    ) -> List[Conflict]:
        """
        数值不一致检测

        逻辑:
          1. 从每条证据内容中提取数值（百分比、普通数值）及其指标名称
          2. 按指标名称分组
          3. 同一指标出现不同数值时，判定为冲突

        指标名称来源（优先级）:
          a. metadata 中的 metric_name 字段
          b. 数值前文中的指标关键词前缀（如"核心一级资本充足率不得低于5%"）

        Args:
            evidence_items: EvidenceItem 列表

        Returns:
            数值不一致冲突列表
        """
        conflicts: List[Conflict] = []

        # metric_key -> [(evidence_id, value_str, value_float), ...]
        metric_map: Dict[str, List[Tuple[str, str, float]]] = {}

        for ev in evidence_items:
            for metric_key, value_str, value_float in self._extract_numeric_metrics(ev):
                metric_map.setdefault(metric_key, []).append(
                    (ev.evidence_id, value_str, value_float)
                )

        for metric_key, entries in metric_map.items():
            # 同一指标至少有两条证据
            if len(entries) < 2:
                continue
            # 取去重后的数值
            distinct_values = sorted(set(v for _, _, v in entries))
            if len(distinct_values) < 2:
                continue  # 数值相同，无冲突

            involved_ids = list(dict.fromkeys(eid for eid, _, _ in entries))
            conflict = Conflict(
                conflict_id=f"conflict-{uuid.uuid4().hex[:8]}",
                conflict_type=ConflictType.NUMERIC_MISMATCH,
                description=(
                    f"指标「{metric_key}」在不同证据中出现不一致数值: "
                    f"{', '.join(set(vs for _, vs, _ in entries))}"
                ),
                evidence_ids=involved_ids,
                details={
                    "metric": metric_key,
                    "values": [
                        {"evidence_id": eid, "value": vs}
                        for eid, vs, _ in entries
                    ],
                },
                priority=CONFLICT_PRIORITY[ConflictType.NUMERIC_MISMATCH],
            )
            conflicts.append(conflict)

        return conflicts

    def _detect_version_conflict(
        self, evidence_items: List[EvidenceItem]
    ) -> List[Conflict]:
        """
        版本冲突检测

        逻辑:
          1. 按 source_doc 分组
          2. 同一文档同时存在 active 和 superseded 版本状态，且内容不同时判定为冲突

        Args:
            evidence_items: EvidenceItem 列表

        Returns:
            版本冲突列表
        """
        conflicts: List[Conflict] = []

        doc_groups = self._group_by_doc(evidence_items)

        for doc_name, items in doc_groups.items():
            if len(items) < 2:
                continue
            statuses = set(ev.version_status for ev in items)
            # 同时存在现行有效与已被替代的版本
            if "active" in statuses and "superseded" in statuses:
                contents = set((ev.content or "").strip() for ev in items)
                # 内容不同才构成冲突（内容相同仅状态差异由去重处理）
                if len(contents) > 1:
                    conflict = Conflict(
                        conflict_id=f"conflict-{uuid.uuid4().hex[:8]}",
                        conflict_type=ConflictType.VERSION_CONFLICT,
                        description=(
                            f"文档「{doc_name}」存在版本状态冲突: "
                            f"同时包含 {', '.join(sorted(statuses))} 版本"
                        ),
                        evidence_ids=[ev.evidence_id for ev in items],
                        details={
                            "source_doc": doc_name,
                            "version_statuses": sorted(statuses),
                            "versions": [
                                {
                                    "evidence_id": ev.evidence_id,
                                    "version_status": ev.version_status,
                                    "citation": ev.citation,
                                }
                                for ev in items
                            ],
                        },
                        priority=CONFLICT_PRIORITY[ConflictType.VERSION_CONFLICT],
                    )
                    conflicts.append(conflict)

        return conflicts

    def _detect_scope_overlap(
        self, evidence_items: List[EvidenceItem]
    ) -> List[Conflict]:
        """
        适用范围重叠检测

        逻辑:
          1. 为每条证据计算适用范围键（metadata.applicable_scope > hierarchy_path 根节点 > source_doc）
          2. 两两比较不同文档的证据
          3. 适用范围键相同或互为子串时，判定为范围重叠

        Args:
            evidence_items: EvidenceItem 列表

        Returns:
            适用范围重叠冲突列表
        """
        conflicts: List[Conflict] = []
        n = len(evidence_items)

        for i in range(n):
            ev_a = evidence_items[i]
            scope_a = self._get_scope_key(ev_a)
            if not scope_a:
                continue
            for j in range(i + 1, n):
                ev_b = evidence_items[j]
                # 同一文档不构成范围重叠（由版本冲突检测覆盖）
                if ev_a.source_doc == ev_b.source_doc:
                    continue
                scope_b = self._get_scope_key(ev_b)
                if not scope_b:
                    continue
                if not self._scopes_overlap(scope_a, scope_b):
                    continue
                # 内容不同才标记（相同内容属于冗余，由去重处理）
                if (ev_a.content or "").strip() == (ev_b.content or "").strip():
                    continue

                conflict = Conflict(
                    conflict_id=f"conflict-{uuid.uuid4().hex[:8]}",
                    conflict_type=ConflictType.SCOPE_OVERLAP,
                    description=(
                        f"适用范围重叠: 「{ev_a.source_doc}」与"
                        f"「{ev_b.source_doc}」均覆盖范围「{scope_a}」"
                    ),
                    evidence_ids=[ev_a.evidence_id, ev_b.evidence_id],
                    details={
                        "scope": scope_a,
                        "sources": [
                            {
                                "evidence_id": ev_a.evidence_id,
                                "source_doc": ev_a.source_doc,
                                "hierarchy_path": ev_a.hierarchy_path,
                            },
                            {
                                "evidence_id": ev_b.evidence_id,
                                "source_doc": ev_b.source_doc,
                                "hierarchy_path": ev_b.hierarchy_path,
                            },
                        ],
                    },
                    priority=CONFLICT_PRIORITY[ConflictType.SCOPE_OVERLAP],
                )
                conflicts.append(conflict)

        return conflicts

    def _detect_authority_conflict(
        self, evidence_items: List[EvidenceItem]
    ) -> List[Conflict]:
        """
        效力冲突检测

        逻辑:
          1. 从证据中提取数值指标，按指标名称分组
          2. 同一指标出现不同数值，且涉及证据的 normative_level 不同时判定为冲突

        与数值不一致的区别:
          - 数值不一致关注"值不同"
          - 效力冲突额外关注"效力层级不同"，强调法律效力层面的矛盾

        Args:
            evidence_items: EvidenceItem 列表

        Returns:
            效力冲突列表
        """
        conflicts: List[Conflict] = []

        # metric_key -> [(evidence_item, value_str, value_float), ...]
        metric_map: Dict[str, List[Tuple[EvidenceItem, str, float]]] = {}

        for ev in evidence_items:
            for metric_key, value_str, value_float in self._extract_numeric_metrics(ev):
                metric_map.setdefault(metric_key, []).append((ev, value_str, value_float))

        for metric_key, entries in metric_map.items():
            if len(entries) < 2:
                continue
            distinct_values = set(v for _, _, v in entries)
            if len(distinct_values) < 2:
                continue  # 数值相同，无冲突

            # 涉及证据的效力层级（normative_level）
            levels = set(
                ev.normative_level for ev, _, _ in entries if ev.normative_level
            )
            # 效力层级不同才构成效力冲突
            if len(levels) < 2:
                continue

            involved = list(dict.fromkeys(ev.evidence_id for ev, _, _ in entries))
            conflict = Conflict(
                conflict_id=f"conflict-{uuid.uuid4().hex[:8]}",
                conflict_type=ConflictType.AUTHORITY_CONFLICT,
                description=(
                    f"效力冲突: 指标「{metric_key}」在不同效力层级文件中规定不同"
                    f"（效力层级: {', '.join(sorted(levels))}）"
                ),
                evidence_ids=involved,
                details={
                    "metric": metric_key,
                    "normative_levels": sorted(levels),
                    "values": [
                        {
                            "evidence_id": ev.evidence_id,
                            "value": vs,
                            "normative_level": ev.normative_level,
                            "source_doc": ev.source_doc,
                        }
                        for ev, vs, _ in entries
                    ],
                },
                priority=CONFLICT_PRIORITY[ConflictType.AUTHORITY_CONFLICT],
            )
            conflicts.append(conflict)

        return conflicts

    def _detect_temporal_conflict(
        self, evidence_items: List[EvidenceItem]
    ) -> List[Conflict]:
        """
        时效冲突检测

        逻辑:
          1. 按 source_doc 分组
          2. 同一文档存在多个不同生效日期时判定为冲突（新旧规定过渡期）

        生效日期来源（优先级）:
          a. metadata 中的 effective_date 字段
          b. metadata 中的 生效日期 字段

        Args:
            evidence_items: EvidenceItem 列表

        Returns:
            时效冲突列表
        """
        conflicts: List[Conflict] = []

        doc_groups = self._group_by_doc(evidence_items)

        for doc_name, items in doc_groups.items():
            if len(items) < 2:
                continue

            # 收集每条证据的生效日期
            dated_items: List[Tuple[EvidenceItem, str]] = []
            for ev in items:
                date = self._get_effective_date(ev)
                if date:
                    dated_items.append((ev, date))

            dates = set(d for _, d in dated_items)
            if len(dates) >= 2:
                conflict = Conflict(
                    conflict_id=f"conflict-{uuid.uuid4().hex[:8]}",
                    conflict_type=ConflictType.TEMPORAL_CONFLICT,
                    description=(
                        f"时效冲突: 文档「{doc_name}」存在多个生效日期版本"
                        f"（{', '.join(sorted(dates))}），可能处于新旧规定过渡期"
                    ),
                    evidence_ids=[ev.evidence_id for ev, _ in dated_items],
                    details={
                        "source_doc": doc_name,
                        "effective_dates": sorted(dates),
                        "versions": [
                            {
                                "evidence_id": ev.evidence_id,
                                "effective_date": d,
                                "version_status": ev.version_status,
                                "citation": ev.citation,
                            }
                            for ev, d in dated_items
                        ],
                    },
                    priority=CONFLICT_PRIORITY[ConflictType.TEMPORAL_CONFLICT],
                )
                conflicts.append(conflict)

        return conflicts

    # ============================================================
    # 内部方法 — 数值与指标提取
    # ============================================================

    def _extract_numeric_metrics(
        self, ev: EvidenceItem
    ) -> List[Tuple[str, str, float]]:
        """
        从证据内容中提取数值指标

        对每条证据，返回 [(metric_key, value_str, value_float), ...]:
          - 优先使用 metadata.metric_name 作为指标名
          - 否则从数值前文通过指标关键词切分出指标名

        提取顺序:
          1. 先匹配百分比（8%、8.5%）
          2. 再匹配普通数值，跳过已被百分比覆盖的数字
          3. 跳过年份（4 位数字后紧跟"年"）

        Args:
            ev: EvidenceItem

        Returns:
            指标-数值元组列表
        """
        results: List[Tuple[str, str, float]] = []
        content = ev.content or ""
        metadata = ev.metadata or {}

        # 优先使用 metadata 中声明的指标名
        metric_name = (
            metadata.get("metric_name")
            or metadata.get("指标名称")
            or ""
        )
        metric_name = str(metric_name).strip().lower() if metric_name else ""

        # 先收集百分比匹配区间，避免普通数值重复匹配
        percent_spans: List[Tuple[int, int]] = []
        for m in self._PERCENT_PATTERN.finditer(content):
            value_str = m.group(0).strip()
            value_float = float(m.group(1))
            metric_key = metric_name or self._extract_metric_context(content, m.start())
            if metric_key:
                results.append((metric_key, value_str, value_float))
            percent_spans.append((m.start(), m.end()))

        # 再匹配普通数值
        for m in self._NUMBER_PATTERN.finditer(content):
            # 跳过已被百分比覆盖的数字
            if any(s <= m.start() < e for s, e in percent_spans):
                continue
            # 跳过年份（如 2023年）
            if self._is_year(content, m.end()):
                continue

            value_str = m.group(1)
            value_float = float(value_str)
            metric_key = metric_name or self._extract_metric_context(content, m.start())
            if metric_key:
                results.append((metric_key, value_str, value_float))

        return results

    def _extract_metric_context(self, content: str, number_start: int) -> str:
        """
        从数值前文中提取指标名称作为分组键

        算法:
          1. 取数值前的前缀文本
          2. 查找前缀中最后一个指标关键词（如"不得低于"）
          3. 关键词之前的文本即为指标名候选
          4. 按句段分隔符切分，取最后一段作为指标名
          5. 无关键词时取前缀末尾 12 字符作为上下文

        Args:
            content: 证据全文
            number_start: 数值在内容中的起始位置

        Returns:
            规范化后的指标名（小写），无效时返回空字符串
        """
        prefix = content[:number_start]
        if not prefix:
            return ""

        # 查找前缀中最后一个指标关键词
        best_kw_pos = -1
        for kw in self._METRIC_KEYWORDS:
            pos = prefix.rfind(kw)
            if pos > best_kw_pos:
                best_kw_pos = pos

        if best_kw_pos >= 0:
            metric_text = prefix[:best_kw_pos]
        else:
            # 无关键词时取末尾 12 字符
            metric_text = prefix[-12:] if len(prefix) > 12 else prefix

        # 按句段分隔符切分，取最后一段
        segments = re.split(self._PHRASE_SEPARATORS, metric_text)
        segments = [s for s in segments if s]
        if not segments:
            return ""

        metric = segments[-1].strip()
        # 过长时取末尾片段（避免长前缀干扰指标匹配）
        if len(metric) > 15:
            metric = metric[-12:]
        # 过短或纯数字的指标视为无效
        if len(metric) < 2 or metric.isdigit():
            return ""

        return metric.lower()

    @staticmethod
    def _is_year(content: str, number_end: int) -> bool:
        """
        判断数值后是否紧跟"年"（用于过滤年份误匹配）

        Args:
            content: 证据全文
            number_end: 数值结束位置

        Returns:
            是年份时返回 True
        """
        # 数值为 4 位且后跟"年"
        if number_end < len(content) and content[number_end] == "年":
            return True
        return False

    # ============================================================
    # 内部方法 — 范围与日期辅助
    # ============================================================

    @staticmethod
    def _group_by_doc(
        evidence_items: List[EvidenceItem]
    ) -> Dict[str, List[EvidenceItem]]:
        """按 source_doc 分组证据"""
        groups: Dict[str, List[EvidenceItem]] = {}
        for ev in evidence_items:
            key = ev.source_doc
            groups.setdefault(key, []).append(ev)
        return groups

    @staticmethod
    def _get_scope_key(ev: EvidenceItem) -> str:
        """
        计算证据的适用范围键

        优先级:
          1. metadata 中的 applicable_scope / scope 字段
          2. hierarchy_path 的根节点（第一个层级）
          3. source_doc

        Returns:
            规范化后的范围键（小写），无可用信息时返回空字符串
        """
        metadata = ev.metadata or {}
        scope = (
            metadata.get("applicable_scope")
            or metadata.get("scope")
            or ""
        )
        if scope:
            return str(scope).strip().lower()

        if ev.hierarchy_path:
            sep = " > "
            if sep in ev.hierarchy_path:
                root = ev.hierarchy_path.split(sep)[0]
            else:
                root = ev.hierarchy_path
            root = root.strip()
            if root:
                return root.lower()

        if ev.source_doc:
            return ev.source_doc.strip().lower()

        return ""

    @staticmethod
    def _scopes_overlap(scope_a: str, scope_b: str) -> bool:
        """
        判断两个适用范围是否重叠

        重叠条件:
          - 范围键相同，或
          - 一个范围键是另一个的子串

        Args:
            scope_a: 范围键 A
            scope_b: 范围键 B

        Returns:
            重叠时返回 True
        """
        if not scope_a or not scope_b:
            return False
        if scope_a == scope_b:
            return True
        if scope_a in scope_b or scope_b in scope_a:
            return True
        return False

    @staticmethod
    def _get_effective_date(ev: EvidenceItem) -> str:
        """
        获取证据的生效日期

        优先级:
          1. metadata 中的 effective_date 字段
          2. metadata 中的 生效日期 字段

        Returns:
            生效日期字符串，无则返回空字符串
        """
        metadata = ev.metadata or {}
        date = metadata.get("effective_date") or metadata.get("生效日期") or ""
        return str(date).strip() if date else ""

    # ============================================================
    # 内部方法 — 冲突去重
    # ============================================================

    @staticmethod
    def _deduplicate_conflicts(conflicts: List[Conflict]) -> List[Conflict]:
        """
        去重: 同类型且涉及相同证据集合的冲突只保留第一条

        Args:
            conflicts: 冲突列表

        Returns:
            去重后的冲突列表（保持原始顺序）
        """
        seen: set = set()
        result: List[Conflict] = []
        for c in conflicts:
            key = (c.conflict_type, tuple(sorted(set(c.evidence_ids))))
            if key in seen:
                continue
            seen.add(key)
            result.append(c)
        return result

    def __repr__(self) -> str:
        return "ConflictDetector()"
