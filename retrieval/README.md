# 文旅知识库问答与行程规划系统 · 检索生成域

面向文旅场景的 **RAG 问答 + 行程规划 Agent**：用户既能问景点与文化知识，也能直接拿到一份带往返交通、逐日安排、住宿餐饮与预算的结构化行程，并在地图上看到路线。

| 维度 | 技术选型 |
|---|---|
| 检索 | BGE-M3 稠密向量 + HyDE 多改写 + BM25 三路并行召回 → RRF 融合 → Cross-Encoder 重排 |
| 编排 | LangGraph 多节点图；意图路由 v2（LLM 主导 + 正则护栏 + 确定性降级） |
| 实时工具 | 天气（双源主备）、驾车路线、铁路（自部署 12306-MCP，真实车次与票价）、景点 POI、住宿餐饮、预算估算、交通方式建议 |
| 存储 | Milvus（向量）+ MongoDB（会话/文档）+ Redis（缓存/记忆/限流）+ MinIO（对象） |
| 服务 | FastAPI + SSE 流式输出；三部署单元（import/query/gateway）容器化 |
| 工程 | 离线质量门禁、分层回归评测（L0–L3）、人工抽检回填 |

---

## 一、能力矩阵

| 能力 | 入口 | 说明 |
|---|---|---|
| 文旅对话问答 | `GET /html` | 统一问答图，流式输出，引用知识库证据 |
| 结构化行程规划 | `GET /trip-page` / 对话内规划 | 独立行程子图，输出 Markdown 行程 + 往返交通 + 预算四段 |
| 行程内嵌地图 | 对话/行程页 | 按天 Marker + 同色折线，取景点/住宿真实坐标 |
| 会话历史 | `GET /history-page` | 会话列表、历史消息、地图回溯 |
| 移动端 | `mobile_miniprogram/` | 微信小程序骨架；经独立 BFF 网关（WSS 流式 + JWT + 入站限流）接入 |
| 可观测 | `GET /health`、`GET /registry` | 启动期角色自检；工具/数据源注册表只读暴露 |

## 二、主链路（检索生成）

```
用户提问
   │
   ├─ 时间直答 / 澄清  ──► 短路返回
   │
   ├─ 意图路由 v2 ── 正则护栏 ──► LLM 主判 ──► 降级规则
   │     输出：改写问句 · 数据源策略(kb|web|kb_then_web) · 工具集 · is_plan
   │
   ├─ 规划槽位闸门（对话内规划）── 五项信息（目的地/出发地/天数/偏好/预算）缺项先反问一次
   │
   ├─ 三路并行召回：BGE-M3 稠密 ｜ HyDE 多改写 ｜ BM25
   │     ＋（is_plan 时）实时工具并行路：天气 · 路线 · 铁路 · POI · 住宿餐饮
   │
   ├─ RRF 融合（含工具路汇入）
   ├─ Rerank（动态 topk + 知识库配额 + 抽取式压缩）
   │
   └─ 分支生成：问答答案 ｜ 结构化行程（LangGraph 子图）
```

**意图路由 v2**（`app/rag/common/intent_route_service.py`）：强信号正则短路 → LLM 单次产出改写问句 / 数据源策略 / 工具 / `is_plan` → 确定性守卫（无行程定义词则回落 `is_plan=false`）→ 异常降级到规则路由。`source_policy` 由主图消费为联网门禁，避免"知识库能答的问题仍全量联网"。

## 三、行程规划 Agent

独立 LangGraph 子图（`app/process/trip_plan/agent/`）：

- **前置**：出发地抽取 → 往返交通采集（高铁类走铁路网关、自驾走高德驾车，全程软失败不阻断）
- **并行采集**：天气 · 驾车路线 · 铁路车次票价 · 景点 POI · 住宿餐饮 → 汇总进 Prompt
- **生成**：结构化行程（逐日安排 + 住宿落位 + 餐饮 + 预算四段）
- **规划后并行校验**：POI 核验 与 预算复核（均只依赖初稿，零额外墙钟开销）→ 复核节点

硬约束贯穿全链路：**不编造**——车次号白名单保留（仅真实车次可写）、门票一律不估、缺失金额标注"预估"、易变信息带"以实际为准"。

## 四、评测与质量门禁

**离线门禁**（一条命令守护三道基线）：

```bash
python scripts/run_quality_gate.py --fast     # 编译 + 依赖方向 lint + 纯离线单测
```

**分层回归**（`docs/回归策略.md`）：

| 层级 | 内容 |
|---|---|
| L0 | state 键扫描（秒级，防未声明键污染） |
| L1/L2 | 按影响面选集回归 · judge 判定 |
| L3 | 人工抽检 ≥25%（`make_review_sheet.py` 生成复核单 → `--apply` 回填修正指标） |

评测集 `data/eval_cases.json` 覆盖问答 / 行程 / 澄清 / 负例 / 跨域，并对 plan 类附加**结构断言**（防止"该出行程却走了普通问答"这类 judge 检不出的形态错误）。

## 五、快速开始

```bash
# 1. 依赖（uv；默认组=查询侧共享底座，导入侧加 --extra import）
uv sync

# 2. 配置：复制模板并填入自己的密钥
cp .env.example .env      # LLM / Milvus / Mongo / Redis / 高德 / 天气 / 铁路

# 3. 启动查询服务（默认 8001）
python app/api/http/query_server.py       # 或 run_query_server.bat
```

启动后访问：

| 地址 | 页面 |
|---|---|
| `http://127.0.0.1:8001/html` | 对话问答 |
| `http://127.0.0.1:8001/trip-page` | 结构化行程 |
| `http://127.0.0.1:8001/history-page` | 会话历史 |

容器化：

```bash
docker compose up -d --build              # query:8001 / gateway:8002（导入服务在导入域仓库）
docker compose --profile infra up -d      # 附带 Milvus / Mongo / MinIO / Redis / 12306-MCP
```

## 六、仓库结构（重要）

本仓库是**检索生成域**工作空间，另外两个并列工作空间为：导入域（`RAG_文旅_import`，MinerU + Qwen3-VL 文档导入链路）与共享真源（`RAG_shared_infra`）。

```
app/
├── api/          # 入口层：HTTP 服务、BFF 网关、启动引导
├── process/      # 编排层：LangGraph 主流图 + 行程子图 + 各节点
├── rag/          # 检索生成层：意图路由、召回、重排、生成、行程服务
├── infra/        # 基础设施层：LLM/Milvus/Mongo/MinIO/Redis + 天气/高德/铁路网关
└── shared/       # 共享层：配置、运行时、Schema、客户端、工具函数
data/        # 评测集与评测脚本（分层回归）
docs/        # 架构、模块、部署、评测、回归策略等文档
scripts/     # 质量门禁、依赖方向 lint、数据回填
test/        # 离线单测（多文件断言式，零外部依赖）
mobile_miniprogram/   # 微信小程序骨架
```

**分层与依赖方向**：`api → process → rag → infra / shared`，由 `scripts/check_import_direction.py` 强制校验（禁止反向 import）。

**共享层（`app/rag/common`、`app/infra`、`app/shared`）**：本地开发为目录 junction，单一真源在 `RAG_shared_infra`，因此不随本仓库版本控制。若从远程直接克隆本仓库，需自行准备这三处（复制共享层代码或建立软链接），再运行门禁与启动脚本。

## 七、文档索引

| 文档 | 内容 |
|---|---|
| `docs/项目架构.md` | 分层架构与模块职责 |
| `docs/检索生成节点说明.md` | 主图节点逐个说明 |
| `docs/行程规划融合说明.md` | 行程子图与对话内规划 |
| `docs/评估方案与报告.md` / `docs/回归策略.md` | 评测口径与分层回归 |
| `docs/部署清单.md` / `docs/部署切片与共享层归属方案.md` | 部署单元与配置分区 |
| `docs/网关鉴权与限流设计.md` / `docs/分层防护清单.md` | 移动端网关安全设计 |
| `docs/问题记录与改进清单.md` | 问题台账 |

---

数据与外部服务：Milvus / MongoDB / MinIO / Redis 自部署；天气、地理编码与 POI 走高德与和风；铁路车次票价来自自部署 12306-MCP，均以官方实时数据为准。
