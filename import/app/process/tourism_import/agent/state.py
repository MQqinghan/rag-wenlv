from typing import TypedDict
import copy


class TourismImportGraphState(TypedDict):
    """
    文旅导入图的状态定义，包含所有节点产生和消费的数据字段。
    主体名 item_name（景点/文化主题/游记标题），并携带文旅专属字段。
    """
    task_id: str

    # --- 流程控制标记 ---
    is_md_read_enabled: bool
    is_pdf_read_enabled: bool
    is_table_read_enabled: bool   # 表格（xlsx/xls/csv）分支开关
    is_text_read_enabled: bool    # 纯文本（txt/json/html/docx）分支开关
                                  # 注意：必须在 schema 中声明，LangGraph 会丢弃 schema 外的字段！

    # --- 路径相关 ---
    local_dir: str
    local_file_path: str
    file_title: str
    pdf_path: str
    md_path: str

    # --- 内容数据 ---
    md_content: str
    chunks: list

    # --- 文旅元数据（一次 LLM 抽取多字段） ---
    item_name: str             # 主体名：景点名/文化主题/游记标题
    content_type: str          # 内容类型枚举值
    region: str                # 所属地区
    cultural_theme: str        # 文化主题
    category: str              # 类别标签
    source_type: str           # 来源类型：pdf/md/xlsx/csv
    tourism_meta: dict         # 完整元数据 {item_name, content_type, region, cultural_theme, category, extra}
    forced_content_type: str   # 前端手动指定的内容类型（非空时跳过 LLM 分类，降本）

    # --- 数据库相关 ---
    embeddings_content: list


graph_default_state: TourismImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "is_table_read_enabled": False,
    "is_text_read_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "content_type": "",
    "region": "",
    "cultural_theme": "",
    "category": "",
    "source_type": "",
    "tourism_meta": {},
    "forced_content_type": "",
    "embeddings_content": [],
}


def create_default_state(**overrides) -> TourismImportGraphState:
    """创建默认状态，支持覆盖。"""
    state = copy.deepcopy(graph_default_state)
    state.update(overrides)
    return state


def get_default_state() -> TourismImportGraphState:
    """返回一个新的状态实例，避免全局变量污染。"""
    return copy.deepcopy(graph_default_state)


if __name__ == "__main__":
    from app.shared.runtime.logger import logger

    logger.info("===== tourism_import state 单元测试 =====")
    s1 = create_default_state(task_id="t001", local_file_path="兵马俑.pdf")
    assert s1["task_id"] == "t001"
    assert s1["is_table_read_enabled"] is False
    assert s1["item_name"] == ""
    logger.info(f"默认状态字段数：{len(s1)}")
    logger.info(f"含 forced_content_type：{'forced_content_type' in s1}")
    logger.info(f"含 is_table_read_enabled：{'is_table_read_enabled' in s1}")

    # 覆盖测试
    s2 = create_default_state(task_id="t002", forced_content_type="景点信息")
    assert s2["forced_content_type"] == "景点信息"
    logger.info("===== tourism_import state 测试通过 =====")
