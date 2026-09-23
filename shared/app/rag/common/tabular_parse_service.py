"""
Excel/CSV 表格解析服务：表格 → 行级 chunk。
每行序列化为 "字段名: 值" 的文本，天然是一块完整语义，无需标题切分。
支持 .xlsx/.xls 多 sheet、.csv（utf-8/gbk 编码兜底）。
"""
from pathlib import Path

import pandas as pd

from app.shared.runtime.logger import logger, step_log
from app.rag.common.chunk_config import TABLE_MAX_ROWS, CELL_MAX_LEN


def _df_to_chunks(df: pd.DataFrame, file_title: str, sheet_name: str, source_type: str) -> list[dict]:
    """DataFrame → 行级 chunk 列表。过滤全空行，单元格截断。"""
    df = df.fillna("")
    chunks: list[dict] = []
    for idx, row in df.iterrows():
        lines = [
            f"{col}: {str(row[col])[:CELL_MAX_LEN]}"
            for col in df.columns
            if str(row[col]).strip()
        ]
        if not lines:
            continue  # 跳过全空行
        chunks.append({
            "content": "\n".join(lines),
            "title": f"{sheet_name}-第{idx + 2}行",   # +2 对齐 Excel 实际行号（表头占1行，idx从0起）
            "parent_title": sheet_name,
            "file_title": file_title,
            "part": 0,
            "source_type": source_type,
        })
    return chunks


@step_log("parse_table_to_chunks")
def parse_table_to_chunks(state: dict) -> dict:
    """表格文件 → 行级 chunks。跳过 md 管线（md_content 置空）。"""
    file_path = state.get("local_file_path")
    if not file_path:
        logger.error("表格解析：local_file_path 为空")
        state["chunks"] = []
        return state

    file_title = state.get("file_title") or Path(file_path).stem
    source_type = Path(file_path).suffix.lstrip(".").lower()

    try:
        if source_type == "csv":
            # 中文运营表格常见 GBK 编码，先 utf-8 后 gbk 兜底
            try:
                df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, dtype=str, keep_default_na=False, encoding="gbk")
            state["chunks"] = _df_to_chunks(df, file_title, "sheet1", source_type)[:TABLE_MAX_ROWS]
        else:
            # 多 sheet 逐个处理：sheet 名作为 parent_title
            sheets = pd.read_excel(file_path, sheet_name=None, dtype=str, keep_default_na=False)
            chunks: list[dict] = []
            for sheet_name, df in sheets.items():
                try:
                    chunks.extend(_df_to_chunks(df, file_title, sheet_name, source_type))
                except Exception as e:
                    logger.warning(f"sheet[{sheet_name}]解析失败，跳过该 sheet 继续：{e}")
                    continue
            state["chunks"] = chunks[:TABLE_MAX_ROWS]

        state["md_content"] = ""   # 跳过 md 管线节点的语义
        state["file_title"] = file_title
        logger.info(f"表格解析完成：file={file_title}, source={source_type}, 生成 {len(state['chunks'])} 条 chunk")
        return state
    except Exception as e:
        logger.error(f"表格解析失败 [{file_path}]：{e}", exc_info=True)
        state["chunks"] = []   # 空块会在下游 require_chunks 校验终止该任务，不影响其他文件
        return state


if __name__ == "__main__":
    # 单元测试：构造临时 csv/xlsx 验证解析
    import os
    import tempfile
    from app.shared.runtime.logger import logger

    logger.info("===== tabular_parse_service 单元测试 =====")

    # 测试1：CSV 解析（utf-8）
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("景点名称,等级,门票价格\n兵马俑,5A,120元\n华清池,5A,120元\n,,\n")
        csv_path = f.name
    try:
        state = {"local_file_path": csv_path, "file_title": "景点清单"}
        result = parse_table_to_chunks(state)
        chunks = result.get("chunks", [])
        logger.info(f"测试1 CSV 解析：生成 {len(chunks)} 条 chunk（应=2，过滤了1空行）")
        assert len(chunks) == 2, f"应为2条，实际{len(chunks)}"
        assert "兵马俑" in chunks[0]["content"]
        assert chunks[0]["source_type"] == "csv"
        logger.info(f"测试1 首条 chunk: {chunks[0]['content']}")
    finally:
        os.unlink(csv_path)

    # 测试2：XLSX 多 sheet 解析
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        xlsx_path = f.name
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame({"景点名称": ["大雁塔", "钟楼"], "门票": ["30元", "20元"]}).to_excel(writer, sheet_name="西安", index=False)
            pd.DataFrame({"景点名称": ["外滩", "东方明珠"], "门票": ["免费", "120元"]}).to_excel(writer, sheet_name="上海", index=False)
        state = {"local_file_path": xlsx_path}
        result = parse_table_to_chunks(state)
        chunks = result.get("chunks", [])
        logger.info(f"测试2 XLSX 多 sheet：生成 {len(chunks)} 条 chunk（应=4）")
        assert len(chunks) == 4, f"应为4条，实际{len(chunks)}"
        sheets = {c["parent_title"] for c in chunks}
        assert sheets == {"西安", "上海"}, f"sheet 集合错误: {sheets}"
        logger.info(f"测试2 涉及 sheet: {sheets}")
    finally:
        os.unlink(xlsx_path)

    # 测试3：不存在的文件 → chunks 为空
    state = {"local_file_path": "not_exist.xlsx"}
    result = parse_table_to_chunks(state)
    assert result.get("chunks") == []
    logger.info("测试3 不存在文件正确返回空 chunks")

    logger.info("===== tabular_parse_service 测试通过 =====")
