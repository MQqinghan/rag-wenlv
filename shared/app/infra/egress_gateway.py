# -*- coding: utf-8 -*-
"""
外发网关（Egress Gateway）：导入侧一切外网调用的唯一出口（A2 出口收敛，2026-09-09）。

范式参考：app/infra/amap_gateway.py 的单例门面。A1 档位配置见 app/shared/config/egress_config.py
（IMPORT_EGRESS_MODE = off | mask | full，默认 mask）。

职责（T3 范围）：
1. 档位判定：off 直接拒绝（EgressBlockedError），不发起任何网络请求 →「一键关停外发」成立；
2. 供应商白名单：目标 host 需在 IMPORT_EGRESS_ALLOWED_HOSTS（未配置回落内置默认名单），
   越权直接拒绝（EgressDeniedError）并审计留痕；
3. 审计留痕：每次外发落 logs/egress_audit.jsonl——时间/任务ID/文档/服务/内容类型/字节数/
   内容指纹(sha256 前 8，**不记原文**)/脱敏命中数/结果/耗时；被拦截行为同样留痕；
4. 执行器收口：本工程外发 SDK（langchain ChatOpenAI VL 与文本 LLM）
   只能在网关内执行，业务代码不再直连 openai 客户端 invoke。

mask/full 档执行差异（A3 已落地）：mask 档在网关内对文本消息做正则+词典脱敏
（app/infra/egress_sanitizer）、多模态图片先本地 OCR 预检敏感（命中即拦下，不发起
VL 外发）；full 档原样透传。两者审计字段均上报 mask_hits/mask_categories。

收口点（业务调用改走网关）：
- image_ocr_service.transcribe_image_to_md       → chat_invoke()
- markdown_image_service.summarize_images        → chat_invoke()

off 档各调用点的降级语义（对齐规划 A1）：
- 独立图片 OCR：显式报错终止该任务（import_server 已按失败处理）；
- Markdown 内嵌图：enrich_markdown_images 入口短路，跳过摘要/上传替换，文字照常导入。
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import requests
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from app.shared.config.common import env_str
from app.shared.config.egress_config import import_egress_mode
from app.shared.runtime.logger import PROJECT_ROOT, logger

# ---- 服务与内容类型常量（审计口径统一用） ----
SERVICE_VISION = "vision"      # 视觉理解（图片 OCR / Markdown 内嵌图摘要）
SERVICE_LLM = "llm"            # 文本 LLM（文旅元数据抽取等）
CT_IMAGE = "image"
CT_TEXT = "text"

# ---- 供应商白名单 ----
# 内置默认名单：当前工程外发供应商（VL/LLM 的 DashScope、智谱、MinerU 文档解析）。
# 部署方可用 IMPORT_EGRESS_ALLOWED_HOSTS（逗号分隔）覆写为更严名单。
ALLOWED_HOSTS_ENV = "IMPORT_EGRESS_ALLOWED_HOSTS"
DEFAULT_ALLOWED_HOSTS = ("dashscope.aliyuncs.com", "open.bigmodel.cn", "mineru.net", "*.aliyuncs.com")

# ---- 审计落盘 ----
AUDIT_DIR = PROJECT_ROOT / "logs"
AUDIT_FILE = AUDIT_DIR / "egress_audit.jsonl"
_FINGERPRINT_CHUNK = 128 * 1024  # 大文件指纹采样窗（首尾各 128KB）


class EgressBlockedError(RuntimeError):
    """外发被档位拦截（IMPORT_EGRESS_MODE=off）：未发起任何网络请求。"""


class EgressDeniedError(RuntimeError):
    """外发目标不在供应商白名单：已拒绝并审计留痕。"""


def _file_fingerprint(path: Path) -> str:
    """大文件内容指纹：≤256KB 全量 sha256；更大取首尾 128KB 拼接采样，前 8 位十六进制。"""
    size = path.stat().st_size
    try:
        if size <= 2 * _FINGERPRINT_CHUNK:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            h = hashlib.sha256()
            with path.open("rb") as f:
                h.update(f.read(_FINGERPRINT_CHUNK))
                f.seek(-_FINGERPRINT_CHUNK, os.SEEK_END)
                h.update(f.read(_FINGERPRINT_CHUNK))
            digest = h.hexdigest()
    except OSError:
        return ""
    return digest[:8]


def _resolve_fingerprint(payload_hash: str = "", payload_path: str | Path | None = None) -> str:
    """优先级：显式 payload_hash > payload_path 文件采样指纹。"""
    if payload_hash:
        return str(payload_hash)[:8]
    if payload_path:
        p = Path(payload_path)
        if p.is_file():
            return _file_fingerprint(p)
    return ""


def _data_url_bytes(url: str) -> tuple[bytes, str]:
    """解析 data:[mime];base64,<payload> → (bytes, mime)。非 data URL 返回 (b"", "")。"""
    if not url.startswith("data:"):
        return b"", ""
    try:
        meta, payload = url.split(",", 1)
        mime = meta[len("data:"):].split(";")[0]
        return base64.b64decode(payload), mime or ""
    except Exception:  # noqa: BLE001
        return b"", ""


def _merge_hits(agg: dict[str, int], hits: dict[str, int]) -> None:
    for k, v in hits.items():
        agg[k] = agg.get(k, 0) + v


def _mask_messages(messages: list) -> tuple[list, dict[str, int], dict | None]:
    """
    mask 档消息脱敏（A3）：文本段正则/词典替换；多模态图片先本地 OCR 预检敏感。
    不改动原消息对象（langchain 消息不可变，命中时按类型重建）。
    Returns: (脱敏后消息列表, 命中类别计数, 图片拦截信息 dict|None)
    """
    from app.infra.egress_sanitizer import scan_sensitive_image, sanitize_text

    out: list = []
    agg: dict[str, int] = {}
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            new_text, hits = sanitize_text(content)
            _merge_hits(agg, hits)
            out.append(type(msg)(content=new_text) if hits else msg)
            continue
        if isinstance(content, list):
            new_items: list = []
            changed = False
            for item in content:
                if not isinstance(item, dict):
                    new_items.append(item)
                    continue
                it = dict(item)
                if it.get("type") == "text" and isinstance(it.get("text"), str):
                    new_text, hits = sanitize_text(it["text"])
                    if hits:
                        it["text"] = new_text
                        changed = True
                        _merge_hits(agg, hits)
                elif it.get("type") == "image_url":
                    url = (it.get("image_url") or {}).get("url", "")
                    if isinstance(url, str) and url.startswith("data:"):
                        img_bytes, mime = _data_url_bytes(url)
                        if img_bytes:
                            scan = scan_sensitive_image(img_bytes, mime)
                            if scan.get("blocked"):
                                return out, agg, {
                                    "reason": f"图片命中敏感（本地OCR：{scan.get('hits', {})}）",
                                    "hits": scan.get("hits", {}),
                                }
                new_items.append(it)
            if changed:
                out.append(type(msg)(content=new_items))
            else:
                out.append(msg)
            continue
        out.append(msg)
    return out, agg, None


class EgressGateway:
    """导入侧外发单例门面：档位判定 + 白名单 + 审计 + SDK 执行收口。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    # ---- 只读状态 ----
    @property
    def mode(self) -> str:
        """当前外发档位（off/mask/full），实时读取 env。"""
        return import_egress_mode()

    @property
    def enabled(self) -> bool:
        """外发是否可用（off=False）。"""
        return self.mode != "off"

    def allowed_hosts(self) -> tuple[str, ...]:
        """生效白名单：IMPORT_EGRESS_ALLOWED_HOSTS（逗号分隔）> 内置默认。"""
        raw = env_str(ALLOWED_HOSTS_ENV, "").strip()
        if raw:
            hosts = tuple(h.strip().lower() for h in raw.split(",") if h.strip())
            if hosts:
                return hosts
        return DEFAULT_ALLOWED_HOSTS

    # ---- 审计 ----
    def _audit(self, **fields: object) -> None:
        record = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": self.mode,
            # 以下字段按 A4 口径统一补齐，缺省为空/0
            "task_id": "",
            "document": "",
            "service": "",
            "content_type": "",
            "target_host": "",
            "payload_bytes": 0,
            "fingerprint": "",
            "mask_hits": 0,
            "mask_categories": "",
            "blocked": False,
            "block_reason": "",
            "result": "ok",
            "error": "",
            "elapsed_ms": 0,
        }
        record.update(fields)
        try:
            AUDIT_DIR.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False)
            with self._lock:
                with AUDIT_FILE.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    f.flush()
        except OSError as e:  # noqa: BLE001 - 审计失败不阻断导入主链路
            logger.warning(f"[外发网关] 审计日志写入失败：{e}")

    # ---- 前置校验 ----
    @staticmethod
    def _infer_host_from_client(client) -> str:
        """按 client 的模型名推断供应商 host（与 get_llm_client 的 glm*→智谱 路由一致）。"""
        model = getattr(client, "model_name", None) or getattr(client, "model", "") or ""
        return "open.bigmodel.cn" if str(model).lower().startswith("glm") else "dashscope.aliyuncs.com"

    def _host_allowed(self, host: str) -> bool:
        """白名单匹配：精确匹配，或命中 *.example.com 后缀通配（裸域与任意子域均放行）。"""
        hosts = self.allowed_hosts()
        host = (host or "").strip().lower()
        if host in hosts:
            return True
        for pat in hosts:
            if pat.startswith("*.") and (host == pat[2:] or host.endswith("." + pat[2:])):
                return True
        return False

    def _guard(self, *, service: str, content_type: str, target_host: str,
               document: str, task_id: str, payload_bytes: int,
               payload_hash: str = "", payload_path: str | Path | None = None) -> None:
        """档位 + 白名单前置校验；拦截时审计并抛异常，不发起网络请求。"""
        if not self.enabled:
            reason = "IMPORT_EGRESS_MODE=off 一键关停外发"
            self._audit(service=service, content_type=content_type, target_host=target_host,
                        document=document, task_id=task_id, payload_bytes=payload_bytes,
                        fingerprint=_resolve_fingerprint(payload_hash, payload_path),
                        blocked=True, block_reason=reason, result="blocked")
            raise EgressBlockedError(
                f"[外发网关] {reason}（service={service}，document={document or '-'}）。"
                f"如需外发请将 IMPORT_EGRESS_MODE 置为 mask/full 后重启导入服务。"
            )
        host = (target_host or "").strip().lower()
        if not self._host_allowed(host):
            reason = f"目标 host 不在供应商白名单：{host or '(空)'}"
            self._audit(service=service, content_type=content_type, target_host=host,
                        document=document, task_id=task_id, payload_bytes=payload_bytes,
                        fingerprint=_resolve_fingerprint(payload_hash, payload_path),
                        blocked=True, block_reason=reason, result="blocked")
            raise EgressDeniedError(
                f"[外发网关] {reason}（service={service}）。当前白名单：{list(self.allowed_hosts())}"
            )

    # ---- 执行器：文本/视觉 LLM（ChatOpenAI） ----
    def chat_invoke(
        self,
        client,
        messages: list,
        *,
        service: str,
        content_type: str,
        target_host: str = "",
        document: str = "",
        task_id: str = "",
        payload_bytes: int = 0,
        payload_hash: str = "",
        payload_path: str | Path | None = None,
        parser: object | None = None,
        retry_on_429: bool = False,
        mask_hook: Callable[[list], list] | None = None,
    ) -> str:
        """
        收口 langchain ChatOpenAI 外发调用（VL 与文本 LLM 通用）。

            未命中 off / 白名单时正常外发并审计；mask 档内置 A3 文本脱敏与图片敏感预检，
            mask_hook（可选）仅在内置脱敏后再链式追加自定义消息改写。
            parser 非空时按 (client | parser).invoke 执行（如 StrOutputParser 场景）。
            target_host 可省略，由网关按 client 模型名推断（glm*→智谱，其余→DashScope）。

        Returns:
            str: 模型返回文本（AIMessage.content 或 parser 结果）。
        """
        host = target_host or self._infer_host_from_client(client)
        self._guard(service=service, content_type=content_type, target_host=host,
                    document=document, task_id=task_id, payload_bytes=payload_bytes,
                    payload_hash=payload_hash, payload_path=payload_path)
        msgs = messages
        mask_hits = 0
        mask_cats = ""
        if self.mode == "mask":
            msgs, cat_hits, image_block = _mask_messages(messages)
            if image_block:
                # 图片本地 OCR 命中敏感：不发起 VL 外发，审计 blocked 后拒绝
                reason = f"mask 档敏感预检拦截：{image_block.get('reason', '')}"
                self._audit(service=service, content_type=content_type, target_host=host,
                            document=document, task_id=task_id, payload_bytes=payload_bytes,
                            fingerprint=_resolve_fingerprint(payload_hash, payload_path),
                            mask_hits=sum(image_block.get("hits", {}).values()),
                            mask_categories=",".join(
                                f"{k}:{v}" for k, v in image_block.get("hits", {}).items()),
                            blocked=True, block_reason=reason)
                raise EgressBlockedError(
                    f"[外发网关] {reason}（service={service}，document={document or '-'}）。"
                    f"如需放行请确认内容合规或关闭图片敏感预检（IMAGE_SENSITIVE_GUARD=off）。"
                )
            mask_hits = sum(cat_hits.values())
            mask_cats = ",".join(f"{k}:{v}" for k, v in cat_hits.items())
            if mask_hits:
                logger.info(f"[外发网关] mask 档脱敏 {mask_hits} 处（{mask_cats}），document={document or '-'}")
            if mask_hook is not None:
                msgs = mask_hook(msgs)  # 自定义追加改写（在内置脱敏之后）
        chain = (client | parser) if parser is not None else client
        t0 = time.time()
        try:
            if retry_on_429:
                from app.infra.llm.providers import invoke_llm_with_retry
                raw = invoke_llm_with_retry(chain, msgs, max_retries=3)
            else:
                raw = chain.invoke(msgs)
        except Exception as e:  # noqa: BLE001
            elapsed = round((time.time() - t0) * 1000)
            self._audit(service=service, content_type=content_type, target_host=host,
                        document=document, task_id=task_id, payload_bytes=payload_bytes,
                        fingerprint=_resolve_fingerprint(payload_hash, payload_path),
                        mask_hits=mask_hits, mask_categories=mask_cats,
                        result="error", error=repr(e)[:200], elapsed_ms=elapsed)
            raise
        text = raw.content if not isinstance(raw, str) and hasattr(raw, "content") else raw
        text = "" if text is None else str(text)
        elapsed = round((time.time() - t0) * 1000)
        self._audit(service=service, content_type=content_type, target_host=host,
                    document=document, task_id=task_id, payload_bytes=payload_bytes,
                    fingerprint=_resolve_fingerprint(payload_hash, payload_path),
                    mask_hits=mask_hits, mask_categories=mask_cats,
                    result="ok", elapsed_ms=elapsed)
        return text

    # ---- 执行器：通用 HTTP 外发（MinerU 等文件上传/下载/轮询，requests 收口于此） ----
    def http_request(
        self,
        method: str,
        url: str,
        *,
        service: str = SERVICE_LLM,
        content_type: str = CT_TEXT,
        document: str = "",
        task_id: str = "",
        payload_bytes: int = 0,
        trust_env: bool = False,
        **requests_kwargs,
    ):
        """
        收口任意 HTTP 外发（MinerU 文档解析的上传/下载/轮询等）。

        method/url 之外的请求参数（headers/json/data/timeout 等）经 **requests_kwargs 透传。
        前置档位 + 白名单（支持 *.aliyuncs.com 后缀通配）校验，命中后审计留痕（含 task_id），
        再发起请求；非 2xx 仅按 error 审计、不抛异常（由调用方判状态码）。
        """
        from urllib.parse import urlparse as _urlparse
        host = (_urlparse(url).hostname or "").lower()
        self._guard(service=service, content_type=content_type, target_host=host,
                    document=document, task_id=task_id, payload_bytes=payload_bytes)
        t0 = time.time()
        session = requests.Session()
        session.trust_env = trust_env
        try:
            resp = session.request(method, url, **requests_kwargs)
        except Exception as e:  # noqa: BLE001
            elapsed = round((time.time() - t0) * 1000)
            self._audit(service=service, content_type=content_type, target_host=host,
                        document=document, task_id=task_id, payload_bytes=payload_bytes,
                        result="error", error=repr(e)[:200], elapsed_ms=elapsed)
            raise
        elapsed = round((time.time() - t0) * 1000)
        self._audit(service=service, content_type=content_type, target_host=host,
                    document=document, task_id=task_id, payload_bytes=payload_bytes,
                    result="ok" if resp.status_code < 400 else "error",
                    error="" if resp.status_code < 400 else f"status={resp.status_code}",
                    elapsed_ms=elapsed)
        return resp

    # ---- 查询：最近审计记录（导入页「外发清单」面板数据源） ----
    def recent_records(self, limit: int = 50, document: str = "") -> list[dict]:
        """从 egress_audit.jsonl 尾部读最近记录（新→旧），可过滤 document 子串。"""
        if not AUDIT_FILE.is_file():
            return []
        try:
            with AUDIT_FILE.open("r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return []
        records: list[dict] = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if document and document not in rec.get("document", ""):
                continue
            records.append(rec)
            if len(records) >= limit:
                break
        return records

    def clear_records(self) -> int:
        """清空审计文件（保留文件）。返回删除行数。"""
        n = 0
        if AUDIT_FILE.is_file():
            with AUDIT_FILE.open("r", encoding="utf-8") as f:
                n = sum(1 for _ in f)
            AUDIT_FILE.write_text("", encoding="utf-8")
        return n


# ---- 进程内单例（与 amap_gateway 门面同款） ----
egress_gateway = EgressGateway()


def audit_host_of_url(url: str | None) -> str:
    """从 URL 提取 host（供业务侧 target_host 计算/日志用）。"""
    if not url:
        return ""
    try:
        return urlparse(url).hostname or ""
    except ValueError:
        return ""
