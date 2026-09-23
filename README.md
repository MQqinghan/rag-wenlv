# 文旅知识库问答与行程规划系统（RAG 检索 + LangGraph Agent）

面向文旅场景的智能问答与行程规划系统：既能回答景点与人文知识，也能直接产出带往返交通、逐日安排、住宿餐饮与预算的结构化行程，并在地图上呈现路线。

本仓库是**发布快照**，按三个工作空间分目录收录，便于完整分发与阅读：

| 目录 | 对应工作空间 | 内容 |
|---|---|---|
| `retrieval/` | 检索生成域 | LangGraph 问答主图 + 行程规划子图、三路并行召回 + RRF + 重排、真实时工具链、意图路由、评测体系与质量门禁、移动端 BFF |
| `import/` | 数据导入域 | MinerU 文档解析 + Qwen3-VL 图片理解、三级切分、BGE-M3 混合向量、Milvus 入库、外发治理（egress 网关） |
| `shared/` | 共享真源 | 跨域共用基础设施：`app/shared`（配置/运行时/Schema）+ `app/infra`（LLM/向量库/对象存储/天气·高德·铁路网关）+ `app/rag/common`（通用服务） |

## 目录结构为什么是这样

三个工作空间在本地是**并列的独立 Git 仓库**，共享层以目录软链接（Windows junction）被两个域引用，保证「单一真源，不做物理复制」。本仓库把三者做成一份只读快照。

因此在本仓库中，`retrieval/app/{shared,infra,rag/common}` 与 `import/app/{shared,infra,rag/common}` 这六个路径**并不存在**——它们在本地开发环境中是指向 `shared/` 的软链接，不纳入版本控制。

要让快照恢复成可运行结构，在仓库根目录执行一次：

```bat
setup_links.cmd
```

脚本会把 `shared/app/...` 链接回两个域（仅创建，不覆盖已存在目录），从而与本地开发环境结构一致。

## 快速开始（以检索域为例）

```bash
# 0.（Windows）重建共享层 junction，使各域 app/ 结构与开发环境一致
setup_links.cmd

# 1. 依赖：进入域目录安装（导入域为 import/，做法相同）
cd retrieval
uv sync

# 2. 配置：复制模板并填写自己的密钥（LLM / Milvus / Mongo / Redis / 高德 / 天气等）
cp .env.example .env

# 3. 启动查询服务（默认 8001）
python app/api/http/query_server.py
```

页面入口（以检索域为例）：

| 地址 | 页面 |
|---|---|
| `http://127.0.0.1:8001/html` | 对话问答 |
| `http://127.0.0.1:8001/trip-page` | 结构化行程 |
| `http://127.0.0.1:8001/history-page` | 会话历史 |

质量门禁（在域目录内执行；编译 + 依赖方向 lint + 离线单测，导入域脚本在 `import/scripts/`）：

```bash
python scripts/run_quality_gate.py --fast
```

检索域更详细的说明见 `retrieval/README.md`。

## 快照更新

本仓库内容由三个工作空间同步而来，如需重新同步（会保留本仓 `.git`）：

```bat
sync_from_workspaces.cmd
```

## 外部依赖

Milvus（向量）、MongoDB（文档/会话）、MinIO（对象存储）、Redis（缓存/限流）、自部署 12306-MCP（铁路车次票价）、高德与和风（地理、POI、天气）。所有实时数据以官方来源为准。
