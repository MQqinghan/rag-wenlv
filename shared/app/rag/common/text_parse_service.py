"""
纯文本/JSON/HTML/DOCX 解析服务：txt/json/html/docx → Markdown 内容，复用既有标题切分管线。
- .txt：utf-8 优先、GBK 兜底读取，包一层一级标题（保证切分时至少产出带标题的块）
- .json：结构化转 Markdown——数组按记录生成小节、对象按顶层 key 生成小节
- .html/.htm：BeautifulSoup 抽正文 → markdownify 转 Markdown，保留标题/段落/列表/表格
- .docx：python-docx 提取标题/段落/列表/表格 → Markdown（二进制格式，不走 _read_text_file）
产物：state["md_content"] + 转换后的 .md 落盘（md_path），供 node_document_split 使用。
"""
import base64
import json
import mimetypes
import re
import urllib.request
from pathlib import Path

from app.shared.runtime.logger import logger, step_log
from app.rag.common.chunk_config import CELL_MAX_LEN, SUPPORTED_IMAGE_EXTENSIONS

TEXT_MAX_CHARS = 1_000_000  # 单文件读取字符上限，防超大文件拖垮服务


def _read_text_file(file_path: str) -> str:
    """读取 txt/json 文本内容：utf-8 优先，GBK 兜底，截断至上限。"""
    try:
        content = Path(file_path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning(f"utf-8 解码失败，尝试 GBK：{file_path}")
        content = Path(file_path).read_text(encoding="gbk")
    return content[:TEXT_MAX_CHARS]


def _json_value_to_lines(value, indent: int = 0) -> list[str]:
    """把 JSON 值转成多行文本：基础类型直接字符串化，嵌套结构 pretty-dump 并截断。"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return [str(value)[:CELL_MAX_LEN]]
    try:
        dumped = json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        dumped = str(value)
    return dumped[:CELL_MAX_LEN].split("\n")


def _json_to_markdown(data, file_title: str) -> str:
    """JSON 数据 → Markdown 文本。数组按记录分节，对象按顶层 key 分节。"""
    lines: list[str] = [f"# {file_title}", ""]

    if isinstance(data, list):
        for idx, record in enumerate(data, start=1):
            lines.append(f"## 记录 {idx}")
            if isinstance(record, dict):
                for key, value in record.items():
                    value_lines = _json_value_to_lines(value)
                    lines.append(f"- {key}: {value_lines[0]}")
                    lines.extend(f"  {v}" for v in value_lines[1:])
            else:
                lines.extend(_json_value_to_lines(record))
            lines.append("")
    elif isinstance(data, dict):
        for key, value in data.items():
            lines.append(f"## {key}")
            lines.extend(_json_value_to_lines(value))
            lines.append("")
    else:
        # 标量 JSON（如纯数字/字符串），整体作为一个小节
        lines.extend(_json_value_to_lines(data))
        lines.append("")

    return "\n".join(lines)


def _save_image_part(part, images_dir: Path, state_box: dict) -> str | None:
    """把 docx 图片 part 落盘到 images_dir，返回 markdown 引用串。

    state_box = {"counter": int, "seen": {partname: ref}} 跨段去重与编号。
    非 SUPPORTED_IMAGE_EXTENSIONS 的 part 跳过（防止把嵌入字体/样式 part 当图）。
    """
    try:
        partname = str(part.partname)  # 形如 /word/media/image1.png
        ext = Path(partname).suffix or ".png"
        if ext.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            return None
        if partname in state_box["seen"]:
            return state_box["seen"][partname]
        images_dir.mkdir(parents=True, exist_ok=True)
        idx = state_box["counter"]
        state_box["counter"] += 1
        name = f"img_{idx}{ext}"
        (images_dir / name).write_bytes(part.blob)
        ref = f"![图{idx}](images/{name})"
        state_box["seen"][partname] = ref
        return ref
    except Exception as e:
        logger.warning(f"docx 图片提取失败：{e}")
        return None


def _extract_docx_images_in_paragraph(para_element, related_parts: dict,
                                       images_dir: Path, state_box: dict) -> list[str]:
    """从段落 XML 抽取所有图片引用（inline drawing 的 a:blip + 旧式 VML v:imagedata）。"""
    from docx.oxml.ns import qn

    refs: list[str] = []
    # 现代 inline/anchor 图片：<w:drawing>/<a:blip r:embed="rId...">（a/r 前缀 python-docx 已注册）
    for blip in para_element.findall(".//" + qn("a:blip")):
        rId = blip.get(qn("r:embed"))
        if not rId:
            continue
        part = related_parts.get(rId)
        if part is None:
            continue
        ref = _save_image_part(part, images_dir, state_box)
        if ref:
            refs.append(ref)
    # 旧式 VML 图片：<v:imagedata r:id="rId...">（v 前缀 python-docx 未注册，用 Clark 全名空间）
    _VML_IMAGEDATA = "{urn:schemas-microsoft-com:vml}imagedata"
    _R_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    for imgdata in para_element.findall(".//" + _VML_IMAGEDATA):
        rId = imgdata.get(_R_ID)
        if not rId:
            continue
        part = related_parts.get(rId)
        if part is None:
            continue
        ref = _save_image_part(part, images_dir, state_box)
        if ref:
            refs.append(ref)
    return refs


def _docx_to_markdown(file_path: str, file_title: str, images_dir: Path | None = None) -> str:
    """
    DOCX → Markdown：python-docx 按文档顺序提取标题/段落/列表/表格。
    二进制格式不走 _read_text_file，直接读文件路径。
    若传入 images_dir，则顺带抽取文档内嵌图片（w:drawing/v:imagedata）落盘到
    images/ 并在对应段落位置插入 ![图N](images/xxx) 引用，供下游 node_md_img 增强。
    """
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as e:
        raise RuntimeError("缺少依赖 python-docx，请先安装：uv pip install python-docx") from e

    doc = Document(file_path)
    related_parts = doc.part.related_parts  # {rId: part}
    lines: list[str] = [f"# {file_title}", ""]
    img_state = {"counter": 0, "seen": {}}  # 跨段编号与去重

    def _table_to_lines(table: Table) -> list[str]:
        """表格 → Markdown 表格（首行作为表头）。"""
        rows = []
        for row in table.rows:
            cells = [cell.text.strip().replace("|", "\\|").replace("\n", " ") for cell in row.cells]
            rows.append(cells)
        if not rows:
            return []
        out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
        for row in rows[1:]:
            out.append("| " + " | ".join(row) + " |")
        return out

    # body 按文档流顺序遍历段落与表格（doc.element.body 迭代保持原始顺序）
    for element in doc.element.body.iterchildren():
        if element.tag.endswith("}p"):
            # 先抽该段内图片引用（即便纯图片段 para.text 为空也要插引用）
            img_refs = (
                _extract_docx_images_in_paragraph(element, related_parts, images_dir, img_state)
                if images_dir is not None else []
            )
            para = Paragraph(element, doc)
            text = para.text.strip()
            if not text and not img_refs:
                lines.append("")
                continue
            if text:
                style_name = (para.style.name or "").lower() if para.style else ""
                if "heading 1" in style_name or style_name == "title":
                    lines.append(f"## {text}")
                elif "heading 2" in style_name:
                    lines.append(f"### {text}")
                elif "heading" in style_name and style_name.split()[-1].isdigit():
                    level = min(int(style_name.split()[-1]) + 1, 6)
                    lines.append(f"{'#' * level} {text}")
                elif "list" in style_name:
                    lines.append(f"- {text}")
                else:
                    lines.append(text)
            # 图片引用紧随该段文本（保持文档流位置）
            for ref in img_refs:
                lines.append(ref)
        elif element.tag.endswith("}tbl"):
            table = Table(element, doc)
            lines.extend(_table_to_lines(table))
            lines.append("")

    md_content = "\n".join(lines)
    md_content = re.sub(r"\n{3,}", "\n\n", md_content).strip()
    img_count = img_state["counter"]
    logger.info(f"DOCX 解析完成：段落+表格转 Markdown，内嵌图片 {img_count} 张，长度={len(md_content)}")
    return md_content


def _resolve_and_save_html_image(src: str, base_dir: str | None,
                                 images_dir: Path, state_box: dict) -> str | None:
    """解析 <img src> 三类来源 → 落盘 images_dir → 返回生成的图片文件名。

    - data: URI → base64 解码
    - 本地相对/绝对路径 → 相对 base_dir 读取（base_dir 为 html 文件父目录）
    - http(s) 远程 → urllib 下载（超时 10s，失败返回 None 跳过）
    失败一律返回 None，调用方保留原 src（下游 scan_images 因无本地文件会自然跳过）。
    """
    try:
        if src.startswith("data:"):
            # data:image/png;base64,XXXX
            header, _, data = src.partition(",")
            if not data:
                return None
            mime = header.split(";")[0]  # data:image/png
            ext = mimetypes.guess_extension(mime.replace("data:", "")) or ".png"
            blob = base64.b64decode(data)
        elif src.startswith(("http://", "https://")):
            req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                blob = resp.read()
                ctype = resp.headers.get_content_type()
                ext = mimetypes.guess_extension(ctype) or Path(src).suffix or ".png"
        else:
            # 本地路径：相对 base_dir 解析
            p = Path(src)
            if not p.is_absolute():
                if not base_dir:
                    return None
                p = Path(base_dir) / src
            if not p.is_file():
                return None
            blob = p.read_bytes()
            ext = p.suffix or ".png"
        if ext.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            return None
        images_dir.mkdir(parents=True, exist_ok=True)
        idx = state_box["counter"]
        state_box["counter"] += 1
        name = f"img_{idx}{ext}"
        (images_dir / name).write_bytes(blob)
        return name
    except Exception as e:
        logger.warning(f"HTML 图片解析失败 [{src[:60]}]：{e}")
        return None


def _html_to_markdown(raw_html: str, file_title: str,
                      base_dir: str | None = None,
                      images_dir: Path | None = None) -> str:
    """
    HTML → Markdown：BeautifulSoup 抽正文（去 script/style/nav/footer 等噪声），
    markdownify 转 Markdown，清理多余空行和首尾空白。
    若传入 images_dir，则处理 <img>（data-uri/本地路径/远程下载）落盘 images/ 并
    重写 src 为 images/xxx，让 markdownify 产出可被下游 node_md_img 命中的引用。
    """
    try:
        from bs4 import BeautifulSoup
        from markdownify import markdownify as md
    except ImportError as e:
        raise RuntimeError(f"HTML 解析依赖缺失：请 pip install markdownify beautifulsoup4：{e}")

    # 1. BeautifulSoup 抽正文，去噪声标签
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup.find_all(["script", "style", "noscript", "nav", "footer", "header", "aside", "iframe"]):
        tag.decompose()

    # 优先 <article> / <main> / <body> 的内容，兜底整个 soup
    container = (
        soup.find("article")
        or soup.find("main")
        or soup.find("body")
        or soup
    )

    # 1.5 图片处理：重写 src 为 images/xxx（markdownify 后产出 ![](images/xxx)）
    img_state = {"counter": 0, "seen": {}}
    if images_dir is not None:
        for img in container.find_all("img"):
            src = (img.get("src") or "").strip()
            if not src:
                continue
            name = _resolve_and_save_html_image(src, base_dir, images_dir, img_state)
            if name:
                img["src"] = f"images/{name}"
            # 解析失败则保留原 src（下游 scan_images 命中不到会自然跳过）

    cleaned_html = str(container)

    # 2. markdownify 转 Markdown
    md_content = md(cleaned_html, heading_style="ATX", bullets="-")

    # 3. 清理：多余空行、首尾空白
    md_content = re.sub(r"\n{3,}", "\n\n", md_content).strip()

    # 4. 从 <title> 或 <h1> 提取标题，否则用文件名
    extracted_title = file_title
    title_tag = soup.find("title")
    h1_tag = soup.find("h1")
    if h1_tag and h1_tag.get_text(strip=True):
        extracted_title = h1_tag.get_text(strip=True)
    elif title_tag and title_tag.get_text(strip=True):
        extracted_title = title_tag.get_text(strip=True)

    # 确保开头有一级标题（如果转换结果已有 # 开头就不重复加）
    if not md_content.startswith("# "):
        md_content = f"# {extracted_title}\n\n{md_content}"

    logger.info(
        f"HTML 解析完成，提取标题=[{extracted_title}]，图片 {img_state['counter']} 张，"
        f"Markdown 长度={len(md_content)}"
    )
    return md_content


@step_log("parse_text_to_markdown")
def parse_text_to_markdown(state: dict) -> dict:
    """txt/json/html 文件 → md_content + 落盘转换后的 .md（写入 local_dir，回填 md_path）。"""
    file_path = state.get("local_file_path")
    if not file_path:
        logger.error("文本解析：local_file_path 为空")
        state["md_content"] = ""
        return state

    file_title = state.get("file_title") or Path(file_path).stem
    source_type = Path(file_path).suffix.lstrip(".").lower()

    try:
        # 图片输出目录：local_dir/images/，与 enrich_markdown_images 的扫描约定一致
        local_dir = state.get("local_dir") or str(Path(file_path).parent)
        images_dir = Path(local_dir) / "images"

        if source_type == "docx":
            # 二进制格式：不走 _read_text_file，直接读文件路径；顺带抽内嵌图片
            md_content = _docx_to_markdown(file_path, file_title, images_dir)
        else:
            raw = _read_text_file(file_path)
            if source_type == "json":
                data = json.loads(raw)
                md_content = _json_to_markdown(data, file_title)
            elif source_type in ("html", "htm"):
                # base_dir 用于解析 <img> 的本地相对路径（相对 html 文件父目录）
                md_content = _html_to_markdown(raw, file_title,
                                               base_dir=str(Path(file_path).parent),
                                               images_dir=images_dir)
            else:  # txt：包一层标题，保证切分器按标题产出带标题块
                md_content = f"# {file_title}\n\n{raw}"

        # 转换后的 .md 落盘：node_document_split 的备份逻辑依赖 md_path
        converted_md_path = Path(local_dir) / f"{file_title}.md"
        converted_md_path.write_text(md_content, encoding="utf-8")

        state["md_content"] = md_content
        state["md_path"] = str(converted_md_path)
        state["file_title"] = file_title
        state["source_type"] = source_type
        logger.info(
            f"文本解析完成：file={file_title}, source={source_type}, "
            f"转换后 md 长度={len(md_content)}, 落盘={converted_md_path}"
        )
        return state
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败 [{file_path}]：{e}")
        state["md_content"] = ""  # 下游切块/入库校验会终止该任务，不影响同批其他文件
        return state
    except Exception as e:
        logger.error(f"文本解析失败 [{file_path}]：{e}", exc_info=True)
        state["md_content"] = ""
        return state


if __name__ == "__main__":
    # 单元测试：txt / json数组 / json对象 / 非法json
    import os
    import tempfile

    logger.info("===== text_parse_service 单元测试 =====")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 测试1：txt
        txt_path = os.path.join(tmp_dir, "云南游记.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("第一段内容。\n第二段内容。")
        state = {"local_file_path": txt_path, "local_dir": tmp_dir}
        result = parse_text_to_markdown(state)
        assert result["md_content"].startswith("# 云南游记")
        assert result["source_type"] == "txt"
        assert Path(result["md_path"]).exists()
        logger.info("测试1 txt 解析通过")

        # 测试2：json 数组
        json_path = os.path.join(tmp_dir, "景点清单.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([{"景点名称": "兵马俑", "等级": "5A"}, {"景点名称": "华清池", "等级": "5A"}], f, ensure_ascii=False)
        state = {"local_file_path": json_path, "local_dir": tmp_dir}
        result = parse_text_to_markdown(state)
        assert "## 记录 1" in result["md_content"]
        assert "- 景点名称: 兵马俑" in result["md_content"]
        assert result["source_type"] == "json"
        logger.info("测试2 json 数组解析通过")

        # 测试3：json 对象
        json_path = os.path.join(tmp_dir, "北欧文化.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"主题": "北欧文化", "特点": ["简约", "自然"]}, f, ensure_ascii=False)
        state = {"local_file_path": json_path, "local_dir": tmp_dir}
        result = parse_text_to_markdown(state)
        assert "## 主题" in result["md_content"]
        logger.info("测试3 json 对象解析通过")

        # 测试4：非法 json → md_content 为空
        bad_path = os.path.join(tmp_dir, "bad.json")
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        state = {"local_file_path": bad_path, "local_dir": tmp_dir}
        result = parse_text_to_markdown(state)
        assert result["md_content"] == ""
        logger.info("测试4 非法 json 正确返回空内容")

    logger.info("===== text_parse_service 测试通过 =====")
