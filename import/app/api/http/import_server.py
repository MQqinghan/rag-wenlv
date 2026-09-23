"""
导入服务 HTTP 入口模块，直接承载导入接口与相关接口业务逻辑。
"""
import os
import shutil
import sys
import threading
import uuid
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path
from typing import List, Dict, Any
from fastapi.responses import FileResponse
# 兼容直接以 `python import_server.py` 方式启动，提前把项目根目录加入模块搜索路径。
if __package__ in (None, ""):
    bootstrap_root = Path(__file__).resolve().parents[3]
    if str(bootstrap_root) not in sys.path:
        sys.path.insert(0, str(bootstrap_root))

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from app.shared.schemas.import_task import ImportStatusResponse, UploadResponse
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.rag.common.chunk_config import SUPPORTED_TEXT_EXTENSIONS, SUPPORTED_TABLE_EXTENSIONS
from app.infra.egress_gateway import egress_gateway
from app.rag.common.image_ocr_service import SUPPORTED_UPLOAD_IMAGE_EXTENSIONS, transcribe_image_to_md
from app.process.tourism_import.agent.main_graph import kb_tourism_import_app
from app.process.tourism_import.agent.state import get_default_state as get_tourism_default_state
from app.infra.config import settings
from app.shared.config.role_config import ROLE_IMPORT, startup_role_check
from app.rag.common.domain_guard_service import inspect_import_guard
from app.rag.common.file_manage_service import list_domain_files, revoke_domain_file
from app.shared.runtime.data_version_service import notify_knowledge_updated
import app.api.bootstrap  # noqa: F401  触发预热/缓存失效钩子注册
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    add_done_task,
    add_running_task,
    clear_task,
    get_done_task_list,
    get_running_task_list,
    get_task_result,
    get_task_status,
    set_task_result,
    update_task_status,
)


app = FastAPI(
    title=settings.import_app_name,
    description="统一 RAG 导入服务（文旅网关分发），负责文件上传、导入执行与状态查询。",
    version="0.3.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins) or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# A5 部署切片（T8）：启动期按角色自检配置，缺失前置到启动暴露（APP_ROLE 未设=本地开发，跳过）
ROLE_CHECK_RESULT = startup_role_check(ROLE_IMPORT)

@app.get("/html")
def import_html():
    """
    返回统一导入演示页面（文旅业务选择）。
    Returns:
        FileResponse: 本地导入演示页面文件响应。
    """
    html_path = PROJECT_ROOT / "app" / "web" / "import.html"
    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])


@app.get("/preview.jpg")
def preview_bg():
    """
    返回页面背景底图（项目根目录的 preview.jpg），供 import.html 做半透明背景使用。
    """
    img_path = PROJECT_ROOT / "preview.jpg"
    return FileResponse(path=img_path, media_type="image/jpeg")


@app.get("/preview2.jpg")
def preview2_bg():
    """
    返回备用背景底图（preview2.jpg），供页面切换背景使用。
    """
    img_path = PROJECT_ROOT / "preview2.jpg"
    return FileResponse(path=img_path, media_type="image/jpeg")


@app.get("/ink-import.webp")
def ink_import_bg():
    """
    返回水墨主题导入页背景底图（png/ 下的灰白水墨山水 webp），供 import.html 水墨主题使用。
    """
    img_path = PROJECT_ROOT / "png" / "5aecd3f8d35eee648d109f0d032b308007bf47127a524-ZBTT9V_fw658.webp"
    return FileResponse(path=img_path, media_type="image/webp")


@app.get("/ink-chat.jpg")
def ink_chat_bg():
    """
    返回水墨主题检索页背景底图（png/ 下的红日红梅水墨 jpg），供 chat.html 水墨主题使用。
    """
    img_path = PROJECT_ROOT / "png" / "7237.jpg_wh860.jpg"
    return FileResponse(path=img_path, media_type="image/jpeg")


@app.get("/ink-scroll.png")
def ink_scroll_img():
    """
    返回导入页页底山水长卷图（png/山水长卷.png：填充2 远山楼阁 + 填充 渔舟湖水 经接缝渐变拼合的一图），
    整幅单块呈现，配诗句题跋与上下轴杆作页面收尾装饰。
    """
    img_path = PROJECT_ROOT / "png" / "山水长卷.png"
    return FileResponse(path=img_path, media_type="image/png")


@app.get("/favicon.ico")
def favicon():
    """
    返回一个简单的内嵌 SVG favicon，避免浏览器自动请求 /favicon.ico 出现 404。
    """
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<rect width="64" height="64" rx="14" fill="#3498db"/>'
        '<text x="32" y="44" font-size="38" text-anchor="middle" fill="#fff">📚</text>'
        "</svg>"
    )
    return Response(content=svg, media_type="image/svg+xml")

# --------------------------
# 领域网关：按用户选择分发到对应导入图（自动识别已移除，必须由用户手选）
# --------------------------
DOMAIN_TOURISM = "tourism"

# 任务挂起状态：领域预检命中"疑似误导"时置为 blocked，等待用户"强制导入/取消"
TASK_STATUS_BLOCKED = "blocked"

_IMPORT_GRAPH_MAP = {
    DOMAIN_TOURISM: (kb_tourism_import_app, get_tourism_default_state),
}

# 导入并发控制：每个文件的全流程含 LLM 元数据抽取 + 向量化，多任务并发会触发 LLM API 限流（429）。
# 同一时刻最多 N 个任务在执行，其余排队（可通过 IMPORT_MAX_CONCURRENT 环境变量调整）。
_IMPORT_CONCURRENCY_SEMAPHORE = threading.Semaphore(int(os.getenv("IMPORT_MAX_CONCURRENT", "3")))


def resolve_import_domain(domain: str, local_file_path: str) -> str:
    """
    解析最终导入领域：仅接受 tourism，非法值直接抛错（任务标记失败），
    不再静默兜底，避免导错库。
    """
    if domain == DOMAIN_TOURISM:
        return domain
    raise ValueError(f"非法导入领域[{domain}]，必须为 tourism（请在前端手动选择）")

# --------------------------
# 后台任务：LangGraph全流程执行
# 独立于主请求线程，由BackgroundTasks触发，避免阻塞接口响应
# --------------------------
def run_graph_task(task_id: str, local_dir: str, local_file_path: str, domain: str = "auto"):
    """
    LangGraph全流程执行后台任务
    核心流程：按领域选择导入图 → 初始化状态 → 流式执行图节点 → 实时更新任务状态 → 异常捕获
    任务状态更新：pending → processing → completed/failed
    节点进度更新：每完成一个节点，将节点名加入done_list，供前端轮询查看

    :param task_id: 全局唯一任务ID，关联单个文件的全流程处理
    :param local_dir: 该任务的本地文件存储目录（含临时文件/解析结果）
    :param local_file_path: 上传文件的本地绝对路径
    :param domain: 导入领域（tourism/auto）
    """
    try:
        # 0. 清掉领域预检挂起时写入的 error（强制放行/重跑时避免残留原因提示）
        set_task_result(task_id, "error", "")
        # 1. 更新任务全局状态为：处理中
        update_task_status(task_id, "processing")
        logger.info(f"[{task_id}] 开始执行LangGraph全流程，本地文件路径：{local_file_path}")

        # 2. 领域分发：auto 时自动识别，否则用用户指定值
        final_domain = resolve_import_domain(domain, local_file_path)
        logger.info(f"[{task_id}] 最终导入领域 -> {final_domain}")
        # 记录实际路由领域，供 /status 返回给前端展示
        set_task_result(task_id, "domain", final_domain)

        # 3. 选择对应领域的导入图与状态工厂
        import_app, state_factory = _IMPORT_GRAPH_MAP[final_domain]

        # 4. 初始化LangGraph状态：加载默认状态 + 注入当前任务的核心参数
        init_state = state_factory()
        init_state["task_id"] = task_id  # 任务ID关联
        init_state["local_dir"] = local_dir  # 任务本地目录
        init_state["local_file_path"] = local_file_path  # 上传文件本地路径

        # 4.5 文旅：注入前端手动指定的内容类型（非空时跳过 LLM 分类，降本）
        if final_domain == DOMAIN_TOURISM:
            forced_content_type = get_task_result(task_id, "forced_content_type", "") or ""
            if forced_content_type:
                init_state["forced_content_type"] = forced_content_type
                logger.info(f"[{task_id}] 注入前端指定内容类型：{forced_content_type}")

        # 5. 流式执行LangGraph全流程（stream模式：实时获取每个节点的执行结果）
        # 信号量限并发：同一时刻最多 IMPORT_MAX_CONCURRENT 个任务在跑，其余在此排队等待，
        # 避免多任务并发打爆 LLM API 速率限制（429）
        logger.info(f"[{task_id}] 等待导入并发额度（当前限额 {os.getenv('IMPORT_MAX_CONCURRENT', '3')}）")
        with _IMPORT_CONCURRENCY_SEMAPHORE:
            logger.info(f"[{task_id}] 获得并发额度，开始执行导入图")
            for event in import_app.stream(init_state):
                for node_name, node_result in event.items():
                    # 记录每个节点完成的日志，包含任务ID和节点名，方便追踪执行顺序
                    logger.info(f"[{task_id}] LangGraph节点执行完成：{node_name}")
                    # 将完成的节点名加入【已完成列表】，前端轮询/status/{task_id}可实时获取
        # 6. 全流程执行完成，更新任务全局状态为：已完成
        update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] LangGraph全流程执行完毕，任务完成")
        # 6.5 缓存失效广播（仅成功路径）：清本进程两件套缓存 + bump Mongo 数据版本号，
        #     查询服务进程在下次查询时经版本比对后自行刷新缓存（两服务为独立进程，无法直接调用）
        notify_knowledge_updated()

    except Exception as e:
        # 7. 捕获全流程异常，更新任务全局状态为：失败，并记录错误日志（含堆栈）
        update_task_status(task_id, "failed")
        # 错误原因写入任务结果：前端轮询 /status/{task_id} 可直接看到失败原因（如文件类型不支持被拒）
        set_task_result(task_id, "error", str(e))
        # 用位置参数传异常文本，避免异常消息中的花括号被 loguru 当占位符解析（KeyError: 'error'）
        logger.error("[{}] LangGraph全流程执行失败，异常信息：{}", task_id, str(e))
        logger.exception(f"[{task_id}] LangGraph全流程执行失败")




# --------------------------
# 核心接口：多文件上传接口（不上传 MinIO）
# 支持多文件批量上传，核心流程：接收文件 → 本地保存 → 启动后台任务
# 访问地址：http://localhost:8000/upload （POST请求，form-data格式传参）
# --------------------------
from pathlib import Path

@app.post("/upload", summary="文件上传接口", description="支持多文件批量上传，按 domain（tourism）分发到对应知识库导入全流程")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    domain: str = Form("auto"),
    forced_content_type: str = Form(""),
):
    """
    文件上传核心接口（不上传 MinIO）
    1. 接收前端上传的多文件（PDF/MD/TXT/JSON/XLSX/CSV）
    2. 按「日期/任务ID」分层保存到本地输出目录，避免文件冲突
    3. 为每个文件生成唯一TaskID，按 domain 分发启动独立的LangGraph后台处理任务
    4. 实时更新任务状态，供前端轮询监控进度

    :param background_tasks: FastAPI后台任务对象，用于异步执行LangGraph流程
    :param files: 前端上传的文件列表（form-data格式）
    :param domain: 导入领域（tourism=文旅）
    :param forced_content_type: 文旅内容类型手动指定（仅 tourism 生效，非空时跳过 LLM 分类）
    :return: 包含上传结果和所有任务ID的JSON响应
    """
    # 1. 构建本地存储根目录：项目根目录/output/YYYYMMDD（按日期分层，方便管理）
    today_str = datetime.now().strftime("%Y%m%d")
    date_based_root_dir: Path = PROJECT_ROOT / "output" / today_str

    # 初始化任务ID列表，用于返回给前端（一个文件对应一个TaskID）
    task_ids = []

    # 支持的全部后缀：pdf/md + 文本类（txt/json/html/docx）+ 表格类（xlsx/xls/csv）
    # + 图片类（jpg/png 等，先经视觉模型转文字再入库）
    supported_all_suffixes = (
        {".pdf", ".md"} | SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_TABLE_EXTENSIONS
        | SUPPORTED_UPLOAD_IMAGE_EXTENSIONS
    )

    # 2. 遍历处理每个上传的文件（多文件批量处理，各自独立生成TaskID）
    for file in files:
        # 生成全局唯一TaskID（UUID4），作为单个文件的全流程标识
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"[{task_id}] 开始处理上传文件，文件名：{file.filename}，文件类型：{file.content_type}")

        # 后缀前置校验：不支持的类型直接标记失败并跳过，不启动导入任务
        upload_suffix = Path(file.filename).suffix.lower()
        if upload_suffix not in supported_all_suffixes:
            logger.warning(f"[{task_id}] 不支持的文件类型[{upload_suffix}]，跳过: {file.filename}")
            set_task_result(
                task_id, "error",
                f"不支持的文件类型: {upload_suffix}"
                f"（支持 PDF/MD/DOCX/TXT/JSON/HTML/XLSX/CSV/JPG/PNG/WEBP/BMP 等图片）",
            )
            update_task_status(task_id, "failed")
            continue

        # 3. 标记「文件上传」阶段为「运行中」，前端轮询可查
        add_running_task(task_id, "upload_file")

        # 4. 构建该任务的本地独立目录：output/YYYYMMDD/TaskID，避免多文件重名冲突
        task_local_dir: Path = date_based_root_dir / task_id
        task_local_dir.mkdir(parents=True, exist_ok=True)

        # 5. 构建上传文件的本地保存绝对路径
        local_file_abs_path: Path = task_local_dir / file.filename

        # 6. 将上传的文件保存到本地临时目录
        with local_file_abs_path.open("wb") as file_buffer:
            shutil.copyfileobj(file.file, file_buffer)
        logger.info(f"[{task_id}] 文件已保存至本地，路径：{local_file_abs_path}")

        # 7. 标记「文件上传」阶段为「已完成」
        add_done_task(task_id, "upload_file")

        # 7.4 图片文件前置转文字：VL 视觉模型把图片转成同目录同名 .md，
        # 之后复用导入图的 md 分支（标题切分→向量化→入库），导入图零改动。
        # 必须放在领域预检之前：blocked 挂起时暂存的已是 md 路径，
        # confirm 强制放行重跑时自然走 md 分支（不会把 .jpg 直接喂给导入图）。
        graph_file_path = local_file_abs_path
        if upload_suffix in SUPPORTED_UPLOAD_IMAGE_EXTENSIONS:
            add_running_task(task_id, "image_ocr")
            try:
                graph_file_path = transcribe_image_to_md(local_file_abs_path, task_id=task_id)
            except Exception as e:  # noqa: BLE001 - 转写失败直接标失败，绝不静默假成功
                logger.error("[{}] 图片转文字失败：{}", task_id, str(e))
                set_task_result(task_id, "error", f"图片识别失败：{e}")
                update_task_status(task_id, TASK_STATUS_FAILED)
                continue
            add_done_task(task_id, "image_ocr")

        # 7.25 保存前端指定的文旅内容类型（强制类型/降本；为空则不指定走 LLM 分类）
        # 存入 task_result：即使命中领域预检挂起，confirm 强制放行重跑时仍可恢复该值
        if domain == DOMAIN_TOURISM and forced_content_type:
            set_task_result(task_id, "forced_content_type", forced_content_type)

        # 7.5 领域误导预检：命中"疑似误导"则挂起，等用户确认后才执行导入（防导错库）
        file_stem = Path(file.filename).stem
        guard_ok, guard_reason = inspect_import_guard(domain, file_stem)
        if not guard_ok:
            # 记录挂起上下文（放行时恢复执行用）与失败原因（/status 展示给用户）
            # 注意存 graph_file_path：图片文件此时已转成 md，放行重跑应继续用 md 路径
            set_task_result(task_id, "local_dir", str(task_local_dir))
            set_task_result(task_id, "local_file_path", str(graph_file_path))
            set_task_result(task_id, "domain", domain)
            set_task_result(task_id, "error", guard_reason)
            update_task_status(task_id, TASK_STATUS_BLOCKED)
            logger.warning(f"[{task_id}] 领域预检命中疑似误导，任务挂起待确认：{guard_reason}")
            continue

        # 8. 将LangGraph全流程处理加入FastAPI后台任务（携带领域参数，供网关分发）
        background_tasks.add_task(
            run_graph_task,
            task_id,
            str(task_local_dir),
            str(graph_file_path),
            domain,
        )
        logger.info(f"[{task_id}] 已将LangGraph全流程加入后台任务，任务已启动")

    # 9. 所有文件处理完毕，返回上传成功信息和所有TaskID
    logger.info(f"多文件上传处理完毕，共处理{len(files)}个文件，生成TaskID列表：{task_ids}，领域：{domain}")
    return UploadResponse(
        code=200,
        message=f"Files uploaded successfully, total: {len(files)}",
        task_ids=task_ids,
        domain=domain,
    )


# --------------------------
# 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
# --------------------------
# 2. 改造接口
@app.get("/status/{task_id}",
         summary="任务状态查询",
         description="根据TaskID查询单个文件的处理进度和全局状态",
         response_model=ImportStatusResponse)  # 绑定模型
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: ImportStatusResponse 格式响应
    """
    # 获取任务各阶段状态
    status = get_task_status(task_id)
    done_list = get_done_task_list(task_id)
    running_list = get_running_task_list(task_id)
    routed_domain = get_task_result(task_id, "domain", "")
    message = get_task_result(task_id, "error", "")  # blocked/failed 的原因提示

    # 记录日志
    logger.info(f"[{task_id}] 任务状态查询，当前状态：{status}，已完成节点：{done_list}")

    return ImportStatusResponse(
        code=200,
        task_id=task_id,
        status=status,
        done_list=done_list,
        running_list=running_list,
        domain=routed_domain,
        message=message,
    )


# --------------------------
# 领域预检人工裁决接口：blocked 任务的「强制导入 / 取消」
# --------------------------
class ImportConfirmRequest(BaseModel):
    action: str = "run"  # run=强制导入 / cancel=取消


@app.post("/upload/confirm/{task_id}",
          summary="领域预检人工裁决",
          description="对 blocked（疑似领域误导）任务执行 强制导入(run) 或 取消(cancel)")
async def confirm_import_task(
    task_id: str,
    request: ImportConfirmRequest,
    background_tasks: BackgroundTasks,
):
    """
    领域预检命中"疑似误导"后，任务被挂起（status=blocked），不会自动导入。
    用户在前端看到原因后可：
    - 强制导入（确认文件确实属于所选领域）：将任务交回后台图流程执行；
    - 取消：删除已暂存的上传文件，清理任务记录。
    """
    current_status = get_task_status(task_id)
    if current_status != TASK_STATUS_BLOCKED:
        return {
            "code": 400,
            "message": f"任务[{task_id}]当前状态[{current_status}]不是待确认(blocked)，无法裁决",
            "task_id": task_id,
            "status": current_status,
        }

    local_dir = get_task_result(task_id, "local_dir", "")
    local_file_path = get_task_result(task_id, "local_file_path", "")
    domain = get_task_result(task_id, "domain", "")

    if request.action == "run":
        if not local_file_path or not os.path.exists(local_file_path):
            return {
                "code": 400,
                "message": f"任务[{task_id}]的暂存文件已丢失，无法强制导入",
                "task_id": task_id,
                "status": current_status,
            }
        # 放行：回到正常后台导入链路（run_graph_task 会自行置 processing/完成状态）
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        background_tasks.add_task(run_graph_task, task_id, local_dir, local_file_path, domain)
        logger.info(f"[{task_id}] 用户确认强制导入（{domain}），任务放行")
        return {
            "code": 200,
            "message": "已确认，开始导入",
            "task_id": task_id,
            "status": TASK_STATUS_PROCESSING,
        }

    if request.action == "cancel":
        # 取消：清理暂存的上传文件与任务记录
        if local_dir and os.path.isdir(local_dir):
            shutil.rmtree(local_dir, ignore_errors=True)
        clear_task(task_id)
        logger.info(f"[{task_id}] 用户取消导入，已清理暂存文件与任务记录")
        return {
            "code": 200,
            "message": "已取消导入并清理暂存文件",
            "task_id": task_id,
            "status": "cancelled",
        }

    return {
        "code": 400,
        "message": f"非法操作[{request.action}]，仅支持 run/cancel",
        "task_id": task_id,
        "status": current_status,
    }


# --------------------------
# 已导入文件管理：列出文件清单 / 撤回文件（不管导入的对不对，都能从知识库删掉）
# 集合名跟随 .env 配置，删除动作委托 file_manage_service（按 file_title 精确匹配两张集合）
# --------------------------
_DOMAIN_LABEL = {DOMAIN_TOURISM: "文旅"}


@app.get("/files",
         summary="已导入文件清单",
         description="列出指定领域知识库中全部已导入文件（按 file_title 聚合，含切块数/主体数）")
async def list_imported_files(domain: str = ""):
    """
    已导入文件清单接口（只读）
    前端"已导入文件管理"面板加载列表用：GET /files?domain=tourism

    :param domain: 导入领域（tourism）
    :return: {"code":200, "domain":..., "files":[{"file_title","chunk_count","entity_count"}]}
    """
    if domain != DOMAIN_TOURISM:
        return {"code": 400, "message": f"非法领域[{domain}]，必须为 tourism", "domain": domain, "files": []}
    try:
        files = list_domain_files(domain)
        logger.info(f"[文件管理] 查询[{domain}]已导入文件清单，共 {len(files)} 个文件")
        return {"code": 200, "message": f"共 {len(files)} 个文件", "domain": domain, "files": files}
    except Exception as e:  # noqa: BLE001 - Milvus 连接异常时给出可读错误而非 500 裸错
        logger.exception(f"[文件管理] 查询[{domain}]文件清单失败")
        return {"code": 500, "message": f"查询失败：{e}", "domain": domain, "files": []}


class RevokeFileRequest(BaseModel):
    domain: str
    file_title: str = ""


@app.post("/files/revoke",
          summary="撤回已导入文件",
          description="按文件名从指定领域知识库删除全部切块与主体索引（不可恢复）")
async def revoke_imported_file(request: RevokeFileRequest):
    """
    撤回已导入文件接口
    不管文件当初导得对不对，只要想撤，就按 file_title 从该域两张 Milvus 集合精确删除。

    :param request: {"domain": "tourism", "file_title": "文件名（不含扩展名）"}
    :return: {"code":200, "message":..., "domain":..., "file_title":..., "detail": {...}}
    """
    if request.domain != DOMAIN_TOURISM:
        return {"code": 400, "message": f"非法领域[{request.domain}]，必须为 tourism"}
    file_title = (request.file_title or "").strip()
    if not file_title:
        return {"code": 400, "message": "file_title 不能为空"}

    domain_label = _DOMAIN_LABEL[request.domain]
    try:
        detail = revoke_domain_file(request.domain, file_title)
        # 汇总删除结果（chunks + entities 两张集合）
        deleted_total = sum(v.get("deleted", 0) for v in detail.values())
        remaining_total = sum(v.get("remaining", 0) for v in detail.values())
        logger.info(f"[文件管理] 撤回[{domain_label}]文件「{file_title}」完成：删除 {deleted_total} 条，残留 {remaining_total} 条")
        if remaining_total > 0:
            message = f"已从{domain_label}知识库撤回「{file_title}」，但仍有 {remaining_total} 条残留，请稍后重试"
        else:
            message = f"已从{domain_label}知识库撤回「{file_title}」"
        return {
            "code": 200,
            "message": message,
            "domain": request.domain,
            "file_title": file_title,
            "detail": detail,
        }
    except Exception as e:  # noqa: BLE001 - 兜底为可读错误
        logger.exception(f"[文件管理] 撤回[{domain_label}]文件「{file_title}」失败")
        return {"code": 500, "message": f"撤回失败：{e}"}


@app.get("/egress/records",
         summary="外发记录（A2 审计面板数据源）",
         description="读取 logs/egress_audit.jsonl 尾部记录（新→旧）；含档位拦截(blocked)行为留痕，不返回原文")
async def egress_records(limit: int = 30, document: str = ""):
    """
    导入侧外发审计查询（只读，供导入页「外发清单」面板轮询）
    审计只记指纹不记原文：时间/文档/服务/内容类型/字节数/指纹/结果/耗时/拦截原因。

    :param limit: 返回条数（默认 30，上限 200）
    :param document: 按文件名子串过滤（可选）
    :return: {"code":200, "mode":"mask", "records":[...]}
    """
    try:
        limit = max(1, min(int(limit), 200))
    except (TypeError, ValueError):
        limit = 30
    try:
        records = egress_gateway.recent_records(limit=limit, document=(document or "").strip())
        return {
            "code": 200,
            "mode": egress_gateway.mode,
            "egress_enabled": egress_gateway.enabled,
            "records": records,
        }
    except Exception as e:  # noqa: BLE001 - 审计查询失败不影响导入主功能
        logger.exception("[外发审计] 查询记录失败")
        return {"code": 500, "message": f"查询失败：{e}", "records": []}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.app_host, port=settings.import_app_port)
