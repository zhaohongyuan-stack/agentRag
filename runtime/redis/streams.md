# Redis Streams 设计

## 用途

Redis Streams 用于异步任务、工具调用结果和状态事件的记录与消费。

## Stream 定义

### 1. 执行事件流 `ace-rag:stream:{session_id}`

记录 Agent 执行过程中的每个状态变更和工具调用。

```
XADD ace-rag:stream:{session_id} * \
  event_type state_transition \
  from_state ROUTED \
  to_state RETRIEVING \
  task_id task-001 \
  timestamp 2026-07-27T10:00:00Z \
  metadata '{"channel":"hybrid","query_id":"q1"}'
```

### 2. 异步检索结果流 `ace-rag:stream:retrieval:{request_id}`

推测式检索的各分支结果通过 Stream 异步返回。

```
XADD ace-rag:stream:retrieval:{request_id} * \
  branch_id branch-1 \
  channel lexical \
  status completed \
  hits_count 5 \
  result_ref redis:evidence:temp:{branch_id}
```

### 3. 工具调用事件流 `ace-rag:stream:tools:{session_id}`

```
XADD ace-rag:stream:tools:{session_id} * \
  tool_name calculator \
  tool_version v1 \
  input_hash abc123 \
  status success \
  duration_ms 50 \
  result_ref redis:tool_result:{call_id}
```

## 消费组

| 消费组 | 消费者 | 用途 |
|--------|--------|------|
| `agent-orchestrator` | DAG 执行器 | 消费检索结果，推进任务 |
| `evidence-builder` | Evidence 模块 | 消费检索结果，组装证据 |
| `audit-logger` | 审计模块 | 持久化所有事件到数据库 |

## 消费策略

- 消费者使用 `XREADGROUP` 阻塞读取。
- 处理完成后 `XACK` 确认。
- 未确认的消息通过 `XPENDING` + `XCLAIM` 重新分配。
- Stream 最大长度通过 `MAXLEN ~` 限制（约 10000 条），旧消息自动淘汰。
