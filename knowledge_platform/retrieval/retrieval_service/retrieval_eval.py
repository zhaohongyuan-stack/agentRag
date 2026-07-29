"""
检索评估 — 黄金集管理与性能基线

功能：
- 加载/管理黄金集（query → relevant_chunk_ids）
- 计算标准检索指标：Recall@K, Precision@K, MRR, NDCG@K
- 各检索阶段耗时基准测试
- 评估报告生成 & 导出

黄金集格式（JSONL，每行一条）：
  {"query": "核心一级资本合格标准", "relevant_chunk_ids": ["c001", "c045"],
   "difficulty": "easy", "category": "资本监管"}

使用方式：
    evaluator = RetrievalEval()
    evaluator.load_golden_set("golden_set.jsonl")
    metrics = evaluator.evaluate(retrieval_api, k_values=[1, 3, 5, 10])
    evaluator.print_report(metrics)
    benchmark = evaluator.benchmark(retrieval_api)
    evaluator.save_report(metrics, benchmark, "report.json")
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable


class RetrievalEval:
    """检索评估器 — 黄金集 + 指标计算 + 基准测试"""

    def __init__(self):
        self.golden_items: List[Dict[str, Any]] = []

    # ============================================================
    # 黄金集管理
    # ============================================================
    def load_golden_set(self, source: str):
        """
        加载黄金集。支持：
        - .jsonl 文件（每行一条）
        - .json 文件（数组）
        """
        source_path = Path(source)
        items = []

        if source_path.suffix == ".jsonl":
            with open(source_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
        elif source_path.suffix == ".json":
            data = json.loads(source_path.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else data.get("items", [])

        # 验证必要字段
        valid = []
        for item in items:
            if item.get("query") and item.get("relevant_chunk_ids"):
                valid.append(item)
            else:
                print(f"  [警告] 跳过无效条目: {item.get('query', 'N/A')[:50]}")

        self.golden_items = valid
        print(f"[RetrievalEval] 已加载黄金集: {len(self.golden_items)} 条查询")

    def add_item(self, query: str, relevant_chunk_ids: List[str],
                 difficulty: str = "medium", category: str = ""):
        """手动添加一条黄金标注"""
        self.golden_items.append({
            "query": query,
            "relevant_chunk_ids": relevant_chunk_ids,
            "difficulty": difficulty,
            "category": category,
        })

    def save_golden_set(self, path: str):
        """导出黄金集到 JSONL 文件"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for item in self.golden_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[RetrievalEval] 黄金集已保存: {p}（{len(self.golden_items)} 条）")

    def stats(self) -> Dict[str, Any]:
        """黄金集统计信息"""
        if not self.golden_items:
            return {"count": 0}
        avg_relevant = sum(len(i["relevant_chunk_ids"]) for i in self.golden_items) / len(self.golden_items)
        difficulties = {}
        for i in self.golden_items:
            d = i.get("difficulty", "unknown")
            difficulties[d] = difficulties.get(d, 0) + 1
        categories = {}
        for i in self.golden_items:
            c = i.get("category", "uncategorized")
            categories[c] = categories.get(c, 0) + 1
        return {
            "count": len(self.golden_items),
            "avg_relevant_per_query": round(avg_relevant, 1),
            "difficulty_dist": difficulties,
            "category_dist": categories,
        }

    # ============================================================
    # 评估指标
    # ============================================================
    def evaluate(self,
                 search_fn: Callable[..., List[Dict[str, Any]]],
                 k_values: List[int] = None,
                 verbose: bool = True) -> Dict[str, Any]:
        """
        对搜索函数进行完整评估。

        参数：
          search_fn:  搜索函数，签名为 (query, top_k) -> [{"chunk_id": ..., "score": ...}, ...]
          k_values:   评估的 K 值列表，如 [1, 3, 5, 10]
          verbose:    是否逐条输出

        返回：
          {"recall": {1: 0.xx, 3: 0.xx, ...},
           "precision": {...}, "mrr": 0.xx, "ndcg": {...},
           "per_query": [...], "summary": "..."}
        """
        k_values = k_values or [1, 3, 5, 10]
        max_k = max(k_values)

        per_query = []
        total_recall = {k: 0.0 for k in k_values}
        total_precision = {k: 0.0 for k in k_values}
        total_ndcg = {k: 0.0 for k in k_values}
        total_mrr = 0.0
        n = 0

        for item in self.golden_items:
            query = item["query"]
            relevant_set = set(item["relevant_chunk_ids"])
            if not relevant_set:
                continue

            # 执行检索
            results = search_fn(query, top_k=max_k)
            retrieved_ids = [r.get("chunk_id", "") for r in results]

            # Recall@K, Precision@K
            recall = {}
            precision = {}
            for k in k_values:
                top_k_ids = retrieved_ids[:k]
                hits = sum(1 for cid in top_k_ids if cid in relevant_set)
                recall[k] = hits / len(relevant_set) if relevant_set else 0.0
                precision[k] = hits / k if k > 0 else 0.0

            # MRR — 第一个相关结果的排名倒数
            mrr = 0.0
            for rank, cid in enumerate(retrieved_ids, 1):
                if cid in relevant_set:
                    mrr = 1.0 / rank
                    break

            # NDCG@K — 归一化折损累积增益
            ndcg = {}
            for k in k_values:
                dcg = 0.0
                idcg = 0.0
                for rank, cid in enumerate(retrieved_ids[:k], 1):
                    rel = 1.0 if cid in relevant_set else 0.0
                    dcg += rel / (__import__('math').log2(rank + 1))
                ideal_rels = sorted([1.0] * len(relevant_set) + [0.0] * (k - len(relevant_set)),
                                    reverse=True)[:k]
                for rank, rel in enumerate(ideal_rels, 1):
                    idcg += rel / (__import__('math').log2(rank + 1))
                ndcg[k] = dcg / idcg if idcg > 0 else 0.0

            per_query.append({
                "query": query,
                "recall": recall,
                "precision": precision,
                "mrr": mrr,
                "ndcg": ndcg,
                "retrieved": retrieved_ids[:max_k],
            })

            for k in k_values:
                total_recall[k] += recall[k]
                total_precision[k] += precision[k]
                total_ndcg[k] += ndcg[k]
            total_mrr += mrr
            n += 1

        if n == 0:
            return {"error": "无有效查询"}

        # 取平均
        avg_recall = {k: round(v / n, 4) for k, v in total_recall.items()}
        avg_precision = {k: round(v / n, 4) for k, v in total_precision.items()}
        avg_ndcg = {k: round(v / n, 4) for k, v in total_ndcg.items()}
        avg_mrr = round(total_mrr / n, 4)

        result = {
            "n_queries": n,
            "k_values": k_values,
            "recall": avg_recall,
            "precision": avg_precision,
            "mrr": avg_mrr,
            "ndcg": avg_ndcg,
            "per_query": per_query,
        }

        if verbose:
            self.print_report(result)

        return result

    def print_report(self, metrics: Dict[str, Any]):
        """打印格式化的评估报告"""
        if "error" in metrics:
            print(f"\n{'='*60}\n评估结果: {metrics['error']}\n{'='*60}")
            return

        print(f"\n{'='*60}")
        print(f"  检索评估报告 — {metrics['n_queries']} 条查询")
        print(f"{'='*60}")

        # 表头
        k_vals = metrics["k_values"]
        header = f"  {'指标':<16}" + "".join(f"  @{k:<6}" for k in k_vals)
        print(header)
        print(f"  {'-'*52}")

        # Recall
        row = f"  {'Recall':<16}" + "".join(f"  {metrics['recall'][k]:<8.4f}" for k in k_vals)
        print(row)

        # Precision
        row = f"  {'Precision':<16}" + "".join(f"  {metrics['precision'][k]:<8.4f}" for k in k_vals)
        print(row)

        # NDCG
        row = f"  {'NDCG':<16}" + "".join(f"  {metrics['ndcg'][k]:<8.4f}" for k in k_vals)
        print(row)

        # MRR
        print(f"  {'MRR':<16}  {metrics['mrr']:.4f}")
        print(f"{'='*60}\n")

        # 按 difficulty 分组的汇总
        if metrics.get("per_query"):
            diff_results = {}
            for pq in metrics["per_query"]:
                d = pq.get("difficulty", "unknown")
                if d not in diff_results:
                    diff_results[d] = {"n": 0, "total_recall_5": 0.0}
                diff_results[d]["n"] += 1
                diff_results[d]["total_recall_5"] += pq["recall"].get(5, 0.0)
            if len(diff_results) > 1:
                print("  按难度分组 Recall@5:")
                for diff, data in sorted(diff_results.items()):
                    avg_r5 = data["total_recall_5"] / data["n"]
                    print(f"    {diff:<10}: {avg_r5:.4f}  ({data['n']} 条)")

    # ============================================================
    # 基准测试
    # ============================================================
    def benchmark(self,
                  search_fn: Callable,
                  queries: List[str] = None,
                  n_runs: int = 3) -> Dict[str, Any]:
        """
        检索各阶段耗时基准。

        返回：
          {"n_queries": int, "n_runs": int,
           "bm25_ms": {"mean": ..., "min": ..., "max": ...},
           "vector_ms": {...}, "total_ms": {...}}
        """
        queries = queries or [item["query"] for item in self.golden_items[:10]]
        if not queries:
            return {"error": "无查询用于基准测试"}

        all_times = []
        for _ in range(n_runs):
            run_times = []
            for q in queries:
                t0 = time.perf_counter()
                results = search_fn(q, top_k=10)
                elapsed = (time.perf_counter() - t0) * 1000  # ms
                run_times.append({"query": q, "total_ms": round(elapsed, 1),
                                  "n_results": len(results)})
            all_times.append(run_times)

        # 汇总统计
        total_times = []
        for run_idx in range(n_runs):
            for t in all_times[run_idx]:
                total_times.append(t["total_ms"])

        def stats(values):
            return {
                "mean": round(sum(values) / len(values), 1),
                "min": round(min(values), 1),
                "max": round(max(values), 1),
                "p50": round(sorted(values)[len(values) // 2], 1),
                "p95": round(sorted(values)[int(len(values) * 0.95)], 1),
            }

        result = {
            "n_queries": len(queries),
            "n_runs": n_runs,
            "total_ms": stats(total_times),
            "detail": [
                {"query": queries[i],
                 "avg_ms": round(sum(run[i]["total_ms"] for run in all_times) / n_runs, 1)}
                for i in range(len(queries))
            ],
        }

        print(f"\n{'='*60}")
        print(f"  检索性能基准 — {len(queries)} 条查询 × {n_runs} 轮")
        print(f"{'='*60}")
        print(f"  平均耗时:  {result['total_ms']['mean']} ms")
        print(f"  最快:      {result['total_ms']['min']} ms")
        print(f"  最慢:      {result['total_ms']['max']} ms")
        print(f"  P50:       {result['total_ms']['p50']} ms")
        print(f"  P95:       {result['total_ms']['p95']} ms")
        print(f"{'='*60}\n")

        return result

    def save_report(self, metrics: Dict[str, Any],
                    benchmark: Dict[str, Any] = None,
                    path: str = "eval_report.json"):
        """保存评估报告到 JSON 文件"""
        report = {
            "golden_set_stats": self.stats(),
            "metrics": {k: v for k, v in metrics.items() if k != "per_query"},
            "per_query": metrics.get("per_query", []),
        }
        if benchmark:
            report["benchmark"] = benchmark

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[RetrievalEval] 报告已保存: {p}")
