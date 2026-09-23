"""
PDF 解析服务模块，负责调用 MinerU 完成 PDF 到 Markdown 的转换。
"""
import shutil
import time
from pathlib import Path

from app.infra.document_parse import mineru_gateway
from app.infra.egress_gateway import SERVICE_LLM, egress_gateway
from app.shared.runtime.logger import logger, step_log
from app.rag.common.chunk_config import (
    MINERU_DOWNLOAD_TIMEOUT_SECONDS,
    MINERU_MODEL_VERSION,
    MINERU_POLL_INTERVAL_SECONDS,
    MINERU_POLL_TIMEOUT_SECONDS,
)
from app.shared.utils.path_util import PROJECT_ROOT


@step_log("validate_pdf_paths")
def validate_pdf_paths(state: dict) -> tuple[Path, Path]:
    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir")
    if not pdf_path:
        logger.error("pdf_path的参数值为空,无法读取文件!")
        raise ValueError("pdf_path的参数值为空,无法读取文件!")
    if not local_dir:
        logger.warning("没有传入local_dir地址,给与默认值!")
        local_dir = PROJECT_ROOT / "output"
        state["local_dir"] = str(local_dir)

    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)
    if not pdf_path_obj.exists():
        logger.error(f"pdf_path:{pdf_path_obj},但是没有文件存在!")
        raise FileNotFoundError(f"pdf_path:{pdf_path_obj},但是没有文件存在!")
    if not local_dir_obj.exists():
        logger.warning(f"local_dir:{local_dir_obj}地址没有文件夹,我们需要主动创建!")
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    return pdf_path_obj, local_dir_obj


@step_log("upload_pdf_and_poll")
def upload_pdf_and_poll(pdf_path_obj: Path, *, task_id: str = "") -> str:
    if not mineru_gateway.base_url or not mineru_gateway.api_key:
        logger.error("minerU配置错误,请检查minerU配置!")
        raise ValueError("minerU配置错误,请检查minerU配置!")

    url = f"{mineru_gateway.base_url}/file-urls/batch"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {mineru_gateway.api_key}",
    }
    payload = {"files": [{"name": pdf_path_obj.stem}], "model_version": MINERU_MODEL_VERSION}

    response = egress_gateway.http_request(
        "POST", url, headers=headers, json=payload,
        service=SERVICE_LLM, document=pdf_path_obj.stem, task_id=task_id, trust_env=False,
    )
    if response.status_code != 200:
        raise RuntimeError(f"申请上传地址失败,返回状态码为:{response.status_code},请检查minerU配置!")
    result_dict = response.json()
    if result_dict["code"] != 0:
        raise RuntimeError(
            f"申请地址网络状态成功!但是业务失败!错误码:{result_dict['code']},失败信息:{result_dict['msg']}"
        )

    file_upload_url = result_dict["data"]["file_urls"][0]
    batch_id = result_dict["data"]["batch_id"]
    upload_response = egress_gateway.http_request(
        "PUT", file_upload_url, data=pdf_path_obj.read_bytes(),
        service=SERVICE_LLM, document=pdf_path_obj.stem, task_id=task_id, trust_env=False,
    )
    if upload_response.status_code != 200:
        raise RuntimeError(f"上传文件失败,返回状态码为:{upload_response.status_code},请检查minerU配置!")

    poll_url = f"{mineru_gateway.base_url}/extract-results/batch/{batch_id}"
    timeout = MINERU_POLL_TIMEOUT_SECONDS
    interval_time = MINERU_POLL_INTERVAL_SECONDS
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError("轮询超时,请检查minerU配置!")
        try:
            poll_response = egress_gateway.http_request(
                "GET", poll_url, headers=headers,
                service=SERVICE_LLM, document=pdf_path_obj.stem, task_id=task_id, trust_env=False,
            )
        except Exception:
            logger.warning("请求出现异常!可以稍后重试!!")
            time.sleep(interval_time)
            continue
        if poll_response.status_code != 200:
            if 500 <= poll_response.status_code < 600:
                logger.warning(f"可有修复的网络异常,状态码为:{poll_response.status_code}")
                time.sleep(interval_time)
                continue
            raise RuntimeError(f"不可修复的网络状态异常,状态码为:{poll_response.status_code}")

        poll_response_dict = poll_response.json()
        if poll_response_dict["code"] != 0:
            raise RuntimeError(
                f"轮询业务异常,错误码:{poll_response_dict['code']},失败信息:{poll_response_dict['msg']}"
            )
        extract_result = poll_response_dict["data"]["extract_result"][0]
        extract_result_state = extract_result["state"]
        if extract_result_state == "done":
            extract_result_url = extract_result["full_zip_url"]
            if not extract_result_url:
                raise RuntimeError("已经完成了解析,但是zip地址为空!!")
            return extract_result_url
        if extract_result_state == "failed":
            raise RuntimeError(f"已经完成了解析,但是失败了!!失败信息:{extract_result['err_msg']}")
        logger.warning(f"解析正在进行中,状态:{extract_result_state}!")
        time.sleep(interval_time)


@step_log("download_and_extract_markdown")
def download_and_extract_markdown(zip_url: str, local_dir_path_obj: Path, stem: str, *, task_id: str = "") -> Path:
    response = egress_gateway.http_request(
        "GET", zip_url, timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS,
        service=SERVICE_LLM, document=stem, task_id=task_id, trust_env=False,
    )
    zip_path_obj = local_dir_path_obj / f"{stem}_result.zip"
    zip_path_obj.write_bytes(response.content)

    extract_path_obj = local_dir_path_obj / stem
    if extract_path_obj.exists():
        shutil.rmtree(extract_path_obj)
    extract_path_obj.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(zip_path_obj, extract_path_obj)

    md_file_list = list(extract_path_obj.rglob("*.md"))
    if not md_file_list:
        raise FileNotFoundError(f"文件解压失败,在:{extract_path_obj}没有任何md文件!")

    for md_file in md_file_list:
        if md_file.stem == stem:
            return md_file

    target_md_obj = None
    for md_file in md_file_list:
        if md_file.name.lower() == "full.md":
            target_md_obj = md_file
            break
    if not target_md_obj:
        target_md_obj = md_file_list[0]
    return target_md_obj.rename(target_md_obj.with_name(f"{stem}.md"))


@step_log("parse_pdf_to_markdown")
def parse_pdf_to_markdown(state: dict) -> dict:
    # 先校验 PDF 路径和输出目录，避免把非法输入送进解析服务。
    task_id = state.get("task_id", "")
    pdf_path_obj, local_dir_path_obj = validate_pdf_paths(state)
    # 上传 PDF 到 MinerU，并轮询直到服务端返回最终压缩包地址。
    zip_url = upload_pdf_and_poll(pdf_path_obj, task_id=task_id)
    logger.info(f"minerU返回的zip地址:{zip_url}")
    # ??????????? Markdown ?????????????
    md_path_obj = download_and_extract_markdown(zip_url, local_dir_path_obj, pdf_path_obj.stem, task_id=task_id)
    state["md_path"] = str(md_path_obj)
    state["md_content"] = md_path_obj.read_text(encoding="utf-8")
    return state
