"""
导入链配置模块，负责集中管理切块、主体识别、图片处理、向量化与入库相关策略参数。
"""


CHUNK_MAX_SIZE = 1000
CHUNK_SIZE = 600
CHUNK_OVERLAP = 20  #文本块之间的重叠字符数，防止关键信息在切分时被截断

ITEM_NAME_CONTEXT_CHUNK_K = 5   # 提取项目名称上下文时，向前/向后关联的文本块数量 (Top-K)
ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS = 10000   # 项目名称上下文拼接后的最大总字符数限制

EMBEDDING_BATCH_SIZE = 5    # 调用 Embedding 模型生成向量时的批次大小（受限于显存或 API 并发）
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

MINERU_MODEL_VERSION = "vlm"    # MinerU 文档解析引擎使用的模型版本（vlm 通常指视觉语言模型，用于复杂版面/图表解析）
MINERU_POLL_TIMEOUT_SECONDS = 600   # MinerU 异步解析任务的最大等待超时时间（秒）
MINERU_POLL_INTERVAL_SECONDS = 3    # 轮询 MinerU 解析任务状态的间隔时间（秒）
MINERU_DOWNLOAD_TIMEOUT_SECONDS = 30     # 从 MinerU 下载解析结果文件的超时时间（秒）

MILVUS_DEFAULT_VARCHAR_MAX_LENGTH = 512 # Milvus 集合中默认 VARCHAR 字段的最大长度
MILVUS_CHUNK_CONTENT_MAX_LENGTH = 65535 # Milvus 中存储的 Chunk 原文内容的最大长度限制
MILVUS_VECTOR_DIM = 1024

# ---- 表格（Excel/CSV）导入参数 ----
SUPPORTED_TABLE_EXTENSIONS = {".xlsx", ".xls", ".csv"}
TABLE_MAX_ROWS = 10000          # 单文件行数上限，防大文件拖垮服务
CELL_MAX_LEN = 2000             # 单元格截断长度

# ---- 纯文本（txt/json/html/docx）导入参数 ----
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".json", ".html", ".htm", ".docx"}
