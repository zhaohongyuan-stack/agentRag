# Redis Keyspace 设计

> B组 Agent 平台的 Redis Key 命名规范

## 命名规范

```
ace-rag:{namespace}:{scope}:{id}
```

| 前缀 | 用途 | TTL |
|------|------|-----|
| `ace-rag:session:{session_id}` | 会话工作状态 | 3600s |
| `ace-rag:queryspec:{session_id}` | 当前 QuerySpec | 随会话 |
| `ace-rag:plan:{query_id}` | 执行计划缓存 | 1800s |
| `ace-rag:evidence:{query_id}` | Evidence Bundle 缓存 | 1800s |
| `ace-rag:dag:{task_id}` | DAG 任务状态 | 随会话 |
| `ace-rag:checkpoint:{session_id}` | 状态机检查点 | 随会话 |
| `ace-rag:budget:{request_id}` | 预算计数器 | 300s |
| `ace-rag:lock:{resource}` | 分布式锁 | 30s |
| `ace-rag:idempotent:{key}` | 幂等键 | 600s |
| `ace-rag:singleflight:{query_hash}` | Single Flight 去重 | 60s |
| `ace-rag:stream:{session_id}` | 执行事件流 | 持久 |
| `ace-rag:ratelimit:{user_id}` | 限流计数 | 60s |
| `ace-rag:cache:embedding:{hash}` | Embedding 缓存 | 86400s |
| `ace-rag:cache:retrieval:{hash}` | 检索结果缓存 | 3600s |
