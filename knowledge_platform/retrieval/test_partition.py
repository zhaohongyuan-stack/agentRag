"""
分区模式检索测试 — 三分库（银行业/保险业/其他）

用法:
    python test_partition.py                              # 内置 4 组路由测试（默认全量，含 Dense 向量）
    python test_partition.py "不良贷款率"                   # CLI 单条查询（默认 hybrid）
    python test_partition.py "偿付能力充足率" -s dense -k 3 -t 保险业
    python test_partition.py -i                            # 交互式连续查询
    python test_partition.py --lightweight                 # 快速模式（跳过 Dense 向量编码）
"""
import argparse
import sys
from pathlib import Path

# 确保能找到 retrieval_service 模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval_service.industry_partition import IndustryPartitionAPI


def build_api(lightweight: bool = False) -> IndustryPartitionAPI:
    """创建并加载分区 API。lightweight=True 跳过 Dense 向量编码（快速冒烟用）。"""
    api = IndustryPartitionAPI(base_dir="regulatory_docs")
    api.load(lightweight=lightweight)
    return api


def _print_hits(hits, max_content: int = 150):
    if not hits:
        print("  (无结果)")
        return
    for h in hits:
        print(f"  [{h.get('rank')}] {h.get('chunk_type')} | "
              f"doc={h.get('doc_name', '')[:30]} | "
              f"score={h.get('score', 0):.4f}")
        content = h.get('content', '')[:max_content].replace('\n', ' ')
        print(f"      {content}")


def run_fixed_tests(api: IndustryPartitionAPI):
    """内置路由回归测试：自动推断 / 显式指定 / 全分区搜索"""
    print("\n" + "=" * 60)
    print("  开始检索测试")
    print("=" * 60)

    test_queries = [
        # (query, filters, target_industries, 说明)
        ("不良贷款率", {"applicable_scope": "商业银行"}, None, "银行业-自动推断"),
        ("偿付能力充足率", {}, ["保险业"], "保险业-显式指定"),
        ("资本充足率", {}, None, "无行业信号-全分区搜索"),
        ("反洗钱", {}, None, "其他行业-自动推断"),
    ]

    for query, filters, target, desc in test_queries:
        print(f"\n{'─' * 60}")
        print(f"  [{desc}] query=\"{query}\"  filters={filters}  target={target}")
        print(f"{'─' * 60}")

        hits = api.search(query, top_k=3, strategy="hybrid",
                          filters=filters, target_industries=target)
        _print_hits(hits)


def run_single(api: IndustryPartitionAPI, query: str, strategy: str,
               top_k: int, target=None, filters=None):
    """单条查询。target=None 时自动从 filters 推断行业，无法推断则全分区搜索。"""
    print(f"\n[单条查询] \"{query}\"  strategy={strategy}  top_k={top_k}  "
          f"target={target or '自动推断/全搜'}  filters={filters or {}}")
    hits = api.search(query, top_k=top_k, strategy=strategy,
                      filters=filters or {}, target_industries=target)
    _print_hits(hits)


def run_interactive(api: IndustryPartitionAPI):
    """交互模式：每行一条查询（hybrid, top_k=5, 行业自动推断/全搜）"""
    print("\n交互模式：每行输入一条查询，Ctrl+C 退出\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        run_single(api, q, "hybrid", 5)


def main():
    parser = argparse.ArgumentParser(
        description="分区模式检索测试（银行业/保险业/其他 三分库）")
    parser.add_argument("query", nargs="?", default=None,
                        help="单条查询文本（不传则跑内置 4 组路由测试）")
    parser.add_argument("-s", "--strategy", default="hybrid",
                        choices=["hybrid", "bm25", "dense", "exact", "metadata", "table"],
                        help="检索策略，默认 hybrid")
    parser.add_argument("-k", "--topk", type=int, default=5,
                        help="返回条数，默认 5")
    parser.add_argument("-t", "--target", nargs="*",
                        choices=["银行业", "保险业", "其他"],
                        help="目标分区（可多个），不传则自动推断/全搜")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="交互式连续查询")
    parser.add_argument("--lightweight", action="store_true",
                        help="跳过 Dense 向量编码（快速冒烟，不测语义检索）")
    args = parser.parse_args()

    api = build_api(lightweight=args.lightweight)

    if args.interactive:
        run_interactive(api)
    elif args.query:
        run_single(api, args.query, args.strategy, args.topk, args.target)
    else:
        run_fixed_tests(api)


if __name__ == "__main__":
    main()
