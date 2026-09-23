"""
回答文本后处理工具：空行压缩与引用标记清理。
answer_output_service（文旅/闲聊）共用。
"""
import re

# 引用标记：模型可能输出【第1块】或【1】【2】【3】两种格式，对用户无意义，统一剥离
_CITATION_MARK_PATTERN = re.compile(r"(?:\s*【(?:第\d+块|\d+)】)+")


def normalize_answer_text(text: str) -> str:
    """
    回答输出前的统一后处理：
    1. 剥离引用标记（【第N块】和【N】两种格式，连续多个一并去掉）
    2. 压缩 2 个以上连续换行为 1 个空行（段落间最多一个空行）
    3. 去除行尾空白与首尾空白
    4. 去除孤立的"图片"文字行（模型误输出图片区块标题但无实际 URL）
    """
    if not text:
        return text
    # 剥离引用标记
    cleaned = _CITATION_MARK_PATTERN.sub("", text)
    # 去除孤立的"图片"/"【图片】"行（模型输出图片标题但无后续 URL）
    cleaned = re.sub(r"^\s*【?图片】?\s*$", "", cleaned, flags=re.MULTILINE)
    # 行尾空白清理
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))
    # 2 个以上连续换行压缩为 2 个（即最多保留一个空行）
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


class BlankLineCompressor:
    """
    流式增量"空行压缩器"：把逐字推送的 delta 中 3 个及以上连续换行压到最多 2 个。

    为什么需要它：normalize_answer_text 只作用于最终完整答案，而流式输出时
    LLM 的原始 delta 会被逐段直接推给前端，模型常输出 \n\n\n\n 这类连续空行，
    用户在打字机渲染过程中就会看到大片空白（实测"？"回答中间出现多条连续空行）。

    实现要点（chunk 边界安全）：
    - 换行符不立即推送，先挂起计数；遇到非换行字符时再把挂起的换行定型吐出。
    - 挂起数封顶 2（一个空行），多余换行永久丢弃。
    - 因此单次 feed() 只可能吐出"已被后续字符定型"的内容，绝不会因未来 delta
      追加换行而需要回溯修改已推送文本。
    - 流结束后残余的尾部换行无需 flush：服务端最终会用 normalize_answer_text
      处理完整答案并通过 SSE FINAL 事件覆盖渲染，尾部空行自然被 strip 掉。
    """

    def __init__(self) -> None:
        self._pending = 0  # 已挂起、尚未定型的连续换行数（0~2）

    def feed(self, delta: str) -> str:
        """喂入一段流式增量，返回其中【可以安全推送】的部分（含已定型空行）。"""
        if not delta:
            return ""
        emitted: list[str] = []
        i, n = 0, len(delta)
        while i < n:
            if delta[i] == "\n":
                self._pending += 1
                if self._pending > 2:
                    self._pending = 2  # 连续换行超过 1 个空行即丢弃多余部分
                i += 1
            else:
                # 遇到非换行字符：挂起的换行定型，先吐出
                if self._pending:
                    emitted.append("\n" * self._pending)
                    self._pending = 0
                # 再连续收集本段剩余的非换行字符
                j = i
                while j < n and delta[j] != "\n":
                    j += 1
                emitted.append(delta[i:j])
                i = j
        return "".join(emitted)
