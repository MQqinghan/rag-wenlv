"""
图片文字识别服务：独立图片文件 → 结构化 Markdown 文本（VL 视觉模型）。

定位：导入服务的前置转换层——图片上传保存后，先经视觉模型转写成同目录同名 .md，
再复用既有导入图的 md 分支（标题切分 → 向量化 → 入库），导入图零改动。
外发收口：VL 调用统一经 app/infra/egress_gateway.chat_invoke（A2），
  受 IMPORT_EGRESS_MODE 档位与供应商白名单约束，并落外发审计（off 档显式报错）。
"""
import base64
import mimetypes
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.infra.egress_gateway import CT_IMAGE, SERVICE_VISION, egress_gateway
from app.infra.llm.providers import llm_provider
from app.shared.runtime.logger import logger

# 支持的图片后缀（导入前置转文字；base64 直传，超上限直接报错不做压缩）。
# 注意与 chunk_config.SUPPORTED_IMAGE_EXTENSIONS（md 内嵌图白名单，含 .gif）区分：
# 本常量管"独立图片文件上传"，gif 为动图不适合转写故不收。
SUPPORTED_UPLOAD_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# 单图原始字节上限：dashscope OpenAI 兼容端点请求体约 10MB，base64 膨胀 4/3，留余量取 7MB
MAX_IMAGE_BYTES = 7 * 1024 * 1024

_OCR_SYSTEM_PROMPT = (
    "你是知识库资料录入员。用户给出一张图片，请把它整理为可入库的 Markdown 资料：\n"
    "1. 原样提取图中全部可读文字（保持原有层级：标题用 #，条目用列表，表格用 Markdown 表格）；\n"
    "2. 末尾加一行「画面描述：…」客观描述画面主体（一到两句）；\n"
    "3. 严禁编造图中不存在的信息；图中无可读文字时只输出画面描述。\n"
    "只输出 Markdown 内容，不要多余解释。"
)


def transcribe_image_to_md(image_path: Path, *, task_id: str = "") -> Path:
    """
    将单张图片经 VL 视觉模型转写为同目录同名 .md 文件。

    :param image_path: 图片本地绝对路径
    :return: 生成的 Markdown 文件路径（同目录同名，仅扩展名不同；file_title 沿用原图文件名）
    :raises ValueError: 图片超过大小上限或视觉模型返回空内容
    :raises Exception: 视觉模型调用或写文件失败（由调用方标记任务失败，不静默假成功）
    """
    size = image_path.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"图片大小 {size / 1024 / 1024:.1f}MB 超过上限 7MB，请压缩后重新上传")

    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    client = llm_provider.vision_chat()
    logger.info(f"[图片识别] 调用视觉模型转写：{image_path.name}（{size / 1024:.0f}KB，{mime}）")
    text = egress_gateway.chat_invoke(
        client,
        [
            SystemMessage(content=_OCR_SYSTEM_PROMPT),
            HumanMessage(content=[
                {"type": "text", "text": "请将这张图片整理为 Markdown 资料。"},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]),
        ],
        service=SERVICE_VISION,
        task_id=task_id,
        content_type=CT_IMAGE,
        document=image_path.name,
        payload_bytes=size,
        payload_path=image_path,
    ).strip()
    if not text:
        raise ValueError("视觉模型返回空内容")

    md_path = image_path.with_suffix(".md")
    md_path.write_text(f"# {image_path.stem}\n\n{text}\n", encoding="utf-8")
    logger.info(f"[图片识别] 转写完成：{md_path.name}（正文 {len(text)} 字）")
    return md_path
