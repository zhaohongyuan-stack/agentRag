# 对接样例数据

本目录为 B 组 Agent 平台提供的 Mock 样例数据，用于在 A 组检索平台未就绪时进行独立开发与联调测试。所有样例数据基于银行业监管法规真实术语构建。

## 目录结构

```
contracts/examples/
├── README.md                          # 本说明文件
├── retrieval_requests/                # 检索请求样例（RetrievalRequest）
│   ├── normal_clause.json             # 正常条款查询
│   ├── normal_threshold.json          # 正常阈值查询
│   ├── normal_table.json              # 正常表格取数
│   ├── normal_definition.json         # 正常定义查询
│   ├── empty_result.json              # 空结果场景
│   ├── timeout.json                   # 超时场景
│   ├── version_conflict.json          # 版本冲突场景
│   └── partial_failure.json           # 部分失败场景
├── retrieval_responses/               # 检索响应样例（RetrievalHit[] / 错误对象）
│   ├── normal_clause.json             # 对应请求：3条clause命中
│   ├── normal_threshold.json          # 对应请求：3条阈值相关命中
│   ├── normal_table.json              # 对应请求：2条cell_fact命中
│   ├── normal_definition.json         # 对应请求：2条glossary命中
│   ├── empty_result.json              # 对应请求：空数组 []
│   ├── timeout.json                   # 对应请求：超时错误对象
│   ├── version_conflict.json          # 对应请求：2个版本（active + superseded）
│   └── partial_failure.json           # 对应请求：部分通道失败，返回降级结果
└── evidence_bundles/                  # EvidenceBundle 样例（B组内部构建）
    ├── sufficient.json                # 证据充分（score=0.92）
    ├── insufficient.json              # 证据不足（score=0.58）
    └── conflict.json                  # 证据冲突（value_conflict）
```

## 一、检索请求样例 (retrieval_requests/)

每个文件是一个 `RetrievalRequest` JSON 对象，对齐 `contracts/schemas/retrieval_request.schema.json`。除标准字段外，额外增加 `scenario` 字段标记场景类型，便于 B 组测试框架区分用例。

| 文件 | 场景 | 查询内容 | 说明 |
|------|------|----------|------|
| `normal_clause.json` | normal | 商业银行资本管理办法第43条 | 按条款号精确过滤，混合检索 |
| `normal_threshold.json` | normal | 核心一级资本充足率最低要求 | 阈值查询，按适用主体过滤 |
| `normal_table.json` | normal | 2026年1月银行业总资产 | 表格取数，strategy=table |
| `normal_definition.json` | normal | 什么是系统重要性银行 | 定义查询，过滤 glossary 类型 |
| `empty_result.json` | empty | 商业银行资本管理办法第一百四十三条 | 查询不存在的条款，返回空 |
| `timeout.json` | timeout | 商业银行资本充足率监管要求汇总 | 大范围检索，标记超时 |
| `version_conflict.json` | version_conflict | 商业银行资本管理办法资本充足率最低要求 | 同一法规多版本命中 |
| `partial_failure.json` | partial_failure | 系统重要性银行附加资本要求与核心一级资本充足率 | 多主题检索，部分通道失败 |

### 请求字段说明

```json
{
  "query": "查询文本",
  "strategy": "hybrid",          // 检索策略：hybrid/bm25/dense/exact/metadata/relation/table
  "top_k": 10,                   // 返回条数
  "filters": {},                 // 元数据过滤（chunk_type/doc_name/clause_number等）
  "scenario": "normal",          // 场景标记（normal/empty/timeout/version_conflict/partial_failure）
  "request_id": "req-xxx-001"    // 请求追踪ID
}
```

## 二、检索响应样例 (retrieval_responses/)

每个文件与同名请求一一对应。正常场景返回 `RetrievalHit[]` 数组（对齐 `contracts/schemas/retrieval_hit.schema.json`），异常场景返回错误对象。

### 关键响应说明

- **normal_clause.json**：3 条 `clause` 类型命中（第四十三/四十四/四十五条），含完整 `scores_detail`（bm25/dense/rrf/rerank）、`matched_by`、`context`（父子兄弟邻域）、`metadata`（含 normative_level=obligatory、applicable_scope、numeric_conditions 等）。
- **normal_table.json**：2 条 `cell_fact` 类型命中。首条 `chunk_id=001_s1_cell_B7`，metadata 含 `row_label=总资产`、`column_label=2026年_1月`、`value=4806061.6912`、`table_name=1. 银行业金融机构`、`unit=亿元`。
- **empty_result.json**：返回空数组 `[]`。
- **timeout.json**：返回错误对象 `{"error": "RT_TIMEOUT", "message": "检索超时"}`，含 elapsed_ms 等诊断信息。
- **version_conflict.json**：返回同一法规 2 个版本（2023年版 active + 2012年版 superseded），metadata 中含 `version_status`、`superseded_by` 字段。
- **partial_failure.json**：返回 2 条降级命中，trace 中含 `channel_status` 与 `failed_channels`，标记 relation 通道失败。

## 三、EvidenceBundle 样例 (evidence_bundles/)

EvidenceBundle 是 B 组内部构建的证据包，将 `RetrievalHit` 转换为回答级证据，对齐 `contracts/schemas/evidence_bundle.schema.json`。

| 文件 | 场景 | sufficiency.score | is_sufficient | 特征 |
|------|------|-------------------|---------------|------|
| `sufficient.json` | 证据充分 | 0.92 | true | 所有 claim_slots 状态为 supported，conflicts 为空 |
| `insufficient.json` | 证据不足 | 0.58 | false | 3 个 claim_slots 为 missing，missing_conditions 列出缺失项 |
| `conflict.json` | 证据冲突 | 0.71 | false | 含 conflicts 数组，type=value_conflict，已 auto_resolved |

### sufficiency.components 说明

| 组件 | 含义 | sufficient | insufficient | conflict |
|------|------|-----------|--------------|----------|
| claim_coverage | 声明覆盖率 | 1.0 | 0.5 | 1.0 |
| source_authority | 来源权威性 | 0.95 | 0.95 | 0.95 |
| version_validity | 版本有效性 | 1.0 | 1.0 | 0.5 |
| condition_completeness | 条件完整性 | 0.9 | 0.4 | 0.8 |
| cross_channel_consistency | 跨通道一致性 | 0.95 | 0.6 | 0.4 |
| conflict_penalty | 冲突惩罚 | 0.0 | 0.0 | 0.3 |
| missing_condition_penalty | 缺失条件惩罚 | 0.0 | 0.25 | 0.0 |

## 四、使用方式

### 1. Mock 检索服务

B 组开发时可将 A 组检索接口替换为 Mock 实现，根据 `request_id` 或 `scenario` 返回对应响应文件：

```python
import json
from pathlib import Path

EXAMPLES_DIR = Path("contracts/examples")

def mock_retrieve(request: dict) -> list | dict:
    scenario = request.get("scenario", "normal")
    # 根据场景映射到响应文件
    response_file = EXAMPLES_DIR / "retrieval_responses" / f"{scenario_or_name}.json"
    return json.loads(response_file.read_text(encoding="utf-8"))
```

### 2. 单元测试

样例数据可直接作为测试夹具（fixture）加载，验证证据组装、冲突检测、充分性评分等模块：

```python
def test_sufficient_bundle():
    bundle = load_json("evidence_bundles/sufficient.json")
    assert bundle["sufficiency"]["is_sufficient"] is True
    assert all(s["status"] == "supported" for s in bundle["claim_slots"])
```

### 3. 联调对照

待 A 组检索平台就绪后，可用相同请求向真实平台发起调用，将返回结果与样例响应做结构对照，验证字段对齐情况。

## 五、数据来源说明

样例数据中引用的法规术语均为真实银行业监管内容，包括：

- 《商业银行资本管理办法》（国家金融监督管理总局 2023 年发布，2024 年 1 月 1 日施行）
- 《系统重要性银行评估办法》（中国人民银行、原银保监会 2020 年发布）
- 银行业金融机构资产负债统计表（中国人民银行按月发布）

> 注：部分条款内容为示意性摘录，实际引用请以法规原文为准。chunk_id、doc_id、sha256 等标识为 Mock 占位值。
