"""
交互式法规检索 — 结构化 RetrievalRequest 输入

输入格式：key=value 对，空格分隔（也叫 "参数风格"）
  - 纯文本（不含 =）→ 自动当作 query
  - 多个字段用空格分隔

示例：
  query=核心一级资本合格标准
  query=不良贷款 strategy=bm25 top_k=10
  query=第十二条 strategy=exact exact_mode=exact
  query=资本 filters=chunk_type:clause,applicable_scope:全部 expand_context=true

RetrievalRequest 支持的全部字段：
  query              - 查询文本（必填）
  strategy           - bm25 | dense | hybrid | exact | metadata | relation | table
  top_k              - 返回条数（默认10）
  bm25_k             - BM25 候选数（默认20）
  vector_k           - 向量候选数（默认20）
  rrf_k              - RRF 融合常数（默认60）
  rerank_k           - 精排候选数（默认30）
  rerank             - 精排模式: none | cross_enc | llm
  exact_mode         - 精确匹配模式: contains | exact | regex | prefix
  expand_context     - 是否扩展邻域: true | false
  include_evidence   - 是否附带出处: true | false
  max_chars_per_hit  - 单条截断上限（默认2000，0=不截断）
  filters            - 元数据过滤: key:value,key:value

快捷命令：
  :q           退出
  :help        查看帮助
  :defaults    查看当前默认值
"""

import shlex
from retrieval_service import (
    RetrievalAPI, RetrievalRequest, RetrievalStrategy, RerankMode
)
from retrieval_service.agent_tool import search_as_tool, to_llm_text, SEARCH_TOOL_SCHEMA


# ── 当前默认值（可运行时修改）──
DEFAULTS = {
    "strategy": "hybrid",
    "top_k": 5,
    "bm25_k": 20,
    "vector_k": 20,
    "rrf_k": 60,
    "rerank_k": 30,
    "rerank": "none",
    "exact_mode": "contains",
    "expand_context": "false",
    "include_evidence": "true",
    "max_chars_per_hit": 2000,
    "filters": "",
}


def parse_input(line: str) -> dict:
    """
    解析 key=value 格式的输入，返回字段 dict。

    特殊处理：
      - 纯文本（不含 =）→ {"query": line}
      - filters=key:val,key:val → {"filters": {"key": "val", ...}}
      - true/false 字符串 → Python bool
    """
    if not line:
        return {}

    # 如果不含 =，当作纯 query
    if "=" not in line:
        return {"query": line}

    params = {}
    # 用 shlex 正确解析带引号的值，例如 query="核心一级 资本"
    try:
        tokens = shlex.split(line)
    except ValueError:
        tokens = line.split()

    for token in tokens:
        if "=" not in token:
            # 裸字符串 → 当作 query
            params["query"] = token
            continue

        key, _, value = token.partition("=")
        key = key.strip().lower()
        value = value.strip().strip("'\"")

        if key == "filters":
            # filters=key1:val1,key2:val2
            fdict = {}
            for part in value.split(","):
                part = part.strip()
                if ":" in part:
                    fk, _, fv = part.partition(":")
                    fdict[fk.strip()] = fv.strip()
            params[key] = fdict
        elif value.lower() == "true":
            params[key] = True
        elif value.lower() == "false":
            params[key] = False
        elif key in ("top_k", "bm25_k", "vector_k", "rrf_k", "rerank_k", "max_chars_per_hit"):
            params[key] = int(value)
        elif key == "rerank":
            params[key] = RerankMode(value) if value else RerankMode.NONE
        elif key == "strategy":
            params[key] = RetrievalStrategy(value)
        else:
            params[key] = value

    return params


def build_request(params: dict) -> RetrievalRequest:
    """合并默认值 + 用户输入，构造 RetrievalRequest"""
    strategy = params.get("strategy") or RetrievalStrategy(DEFAULTS["strategy"])
    rerank = params.get("rerank") or RerankMode(DEFAULTS["rerank"])

    return RetrievalRequest(
        query=params.get("query", ""),
        top_k=params.get("top_k", DEFAULTS["top_k"]),
        strategy=strategy,
        rerank=rerank,
        bm25_k=params.get("bm25_k", DEFAULTS["bm25_k"]),
        vector_k=params.get("vector_k", DEFAULTS["vector_k"]),
        rrf_k=params.get("rrf_k", DEFAULTS["rrf_k"]),
        rerank_k=params.get("rerank_k", DEFAULTS["rerank_k"]),
        exact_mode=params.get("exact_mode", DEFAULTS["exact_mode"]),
        expand_context=params.get("expand_context", DEFAULTS["expand_context"] == "true"),
        include_evidence=params.get("include_evidence", DEFAULTS["include_evidence"] == "true"),
        max_chars_per_hit=params.get("max_chars_per_hit", DEFAULTS["max_chars_per_hit"]),
        filters=params.get("filters", {}),
    )


def print_help():
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  输入格式：key=value key=value ...                       │
  │                                                         │
  │  纯文本自动当作 query（跟搜索引擎一样）                    │
  │                                                         │
  │  示例：                                                  │
  │    核心一级资本                                          │
  │    query=不良贷款 strategy=bm25 top_k=10                 │
  │    query=第十二条 strategy=exact exact_mode=exact         │
  │    query=资本 filters=chunk_type:clause expand_context=true │
  │                                                         │
  │  :q           退出                                      │
  │  :help        查看帮助                                  │
  │  :defaults    查看当前默认值                             │
  │  :set KEY=VALUE  修改默认值                             │
  └─────────────────────────────────────────────────────────┘
""")


def print_defaults():
    print("\n  当前默认值:")
    for k, v in DEFAULTS.items():
        print(f"    {k} = {v}")
    print()


def print_results(hits, verbose=False):
    """打印 RetrievalHit 列表"""
    if not hits:
        print("  (无匹配结果)")
        return

    print(f"\n  📋 共 {len(hits)} 条结果:\n")
    for h in hits:
        if verbose:
            # ── 完整打印所有字段 ──
            print(f"  ╔══ [{h.rank}] ═══════════════════════════════════════")
            print(f"  ║ identity")
            print(f"  ║   chunk_id:       {h.chunk_id}")
            print(f"  ║   chunk_type:     {h.chunk_type}")
            print(f"  ║   doc_id:         {h.doc_id}")
            print(f"  ║   doc_name:       {h.doc_name}")
            print(f"  ║   doc_title:      {h.doc_title}")
            print(f"  ║   hierarchy_path: {h.hierarchy_path}")
            print(f"  ║   source_file:    {h.source_file}")
            print(f"  ╠══ content")
            print(f"  ║   content:        {h.content[:200].replace(chr(10), ' ')}")
            print(f"  ║   evidence_snippet: {h.evidence_snippet}")
            print(f"  ╠══ scores")
            print(f"  ║   score:          {h.score:.4f}")
            print(f"  ║   scores_detail:  {h.scores_detail}")
            print(f"  ║   matched_by:     {h.matched_by}")
            print(f"  ╠══ trace")
            for tk, tv in h.trace.items():
                print(f"  ║   {tk}: {tv}")
            print(f"  ╠══ citation")
            print(f"  ║   {h.citation}")
            if h.context:
                ctx_keys = [k for k, v in h.context.items() if v]
                print(f"  ╠══ context")
                print(f"  ║   keys: {ctx_keys}")
            print(f"  ╚══════════════════════════════════════════════════════")
            print()
        else:
            print(f"  [{h.rank}] {h.citation}")
            detail = "  ".join(f"{k}={v:.4f}" for k, v in h.scores_detail.items())
            print(f"      得分: {h.score:.4f}  |  {detail}")
            print(f"      通道: {h.matched_by}")
            preview = h.content[:150].replace("\n", " ")
            print(f"      内容: {preview}...")
            print()


def main():
    print("正在加载检索引擎...")
    api = RetrievalAPI()
    api.load("regulatory_docs/")
    print(f"加载完成！{api.chunk_count} 个 chunks 已就绪。\n")
    print_help()

    verbose = False
    tool_mode = False

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not line:
            continue

        # ── 快捷命令 ──
        if line == ":q":
            print("再见！")
            break
        if line == ":help":
            print_help()
            continue
        if line == ":defaults":
            print_defaults()
            continue
        if line == ":verbose":
            verbose = not verbose
            print(f"  verbose = {verbose}")
            continue
        if line == ":tool":
            tool_mode = not tool_mode
            print(f"  tool_mode = {tool_mode}  (LLM 视角 {'开' if tool_mode else '关'})")
            continue
        if line == ":schema":
            import json
            print(json.dumps(SEARCH_TOOL_SCHEMA, ensure_ascii=False, indent=2))
            continue
        if line.startswith(":set"):
            _, _, rest = line.partition(" ")
            if "=" in rest:
                k, _, v = rest.partition("=")
                k = k.strip()
                if k in DEFAULTS:
                    DEFAULTS[k] = v.strip()
                    print(f"  {k} → {DEFAULTS[k]}")
                else:
                    print(f"  未知字段: {k}")
            continue

        # ── 解析输入 → 构造 RetrievalRequest → 检索 ──
        params = parse_input(line)
        req = build_request(params)

        print(f"\n  ═══ RetrievalRequest ═══")
        print(f"  query:      {req.query}")
        print(f"  strategy:   {req.strategy.value}")
        print(f"  top_k:      {req.top_k}")
        if req.filters:
            print(f"  filters:    {req.filters}")
        if req.strategy == RetrievalStrategy.EXACT:
            print(f"  exact_mode: {req.exact_mode}")
        if req.expand_context:
            print(f"  expand_context: True")
        if req.rerank != RerankMode.NONE:
            print(f"  rerank:     {req.rerank.value}")
        print()

        hits = api.search_request(req)

        if tool_mode:
            # ── Agent 视角：搜完直接出 LLM 文本 ──
            tool_hits = search_as_tool(
                api, req.query,
                strategy=req.strategy.value,
                top_k=req.top_k,
                filters=req.filters,
            )
            print(to_llm_text(tool_hits))
        else:
            print_results(hits, verbose)


if __name__ == "__main__":
    main()
