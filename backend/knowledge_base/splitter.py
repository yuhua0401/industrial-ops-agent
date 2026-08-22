"""
splitter - 文档智能分块

按章节和语义对已加载的文档进行分块。

本模块只负责“分块”这一单一职责：
- 文档加载见 loader.py
- Contextual RAG 上下文增强见 contextual.py
- 嵌入见 embedder.py
- 完整建库流水线见 scripts/build_knowledge_base.py
"""
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

_settings = get_settings()


# ── 模块级分块器单例 ──────────────────────────────────────────
_MD_HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#",   "H1"),
        ("##",  "H2"),
        ("###", "H3"),
        ("####", "H4"),
    ],
    strip_headers=False,
)

_CHAR_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=_settings.chunk_size_pdf,
    chunk_overlap=_settings.chunk_overlap,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)


# ── PDF 分块 ──────────────────────────────────────────────────

def split_pdf_documents(pages: list[Document]) -> list[Document]:
    """PDF 文档分块：过滤空页 + RecursiveCharacterTextSplitter"""
    non_empty_pages = [p for p in pages if len(p.page_content.strip()) > 20]
    skipped = len(pages) - len(non_empty_pages)
    if skipped > 0:
        logger.info("pdf.split.skip_empty_pages", count=skipped)

    chunks = _CHAR_SPLITTER.split_documents(non_empty_pages)

    for chunk in chunks:
        filename = Path(chunk.metadata.get("source", "未知文件")).stem
        page_num = chunk.metadata.get("page", 0) + 1
        chunk.metadata["source_name"] = f"{filename} 第{page_num}页"

    logger.info("pdf.split.done", pages=len(non_empty_pages), chunks=len(chunks))
    return chunks


# ── Markdown 分块 ─────────────────────────────────────────────

def split_markdown_documents(
    docs: list[Document],
    chunk_size: int = 0,          # 0 = 使用配置默认值
    chunk_overlap: int = 0,       # 0 = 使用配置默认值
) -> list[Document]:
    """Markdown 文档分块：MarkdownHeaderTextSplitter + MarkdownTextSplitter 两阶段"""
    size     = chunk_size     or _settings.chunk_size_md
    overlap  = chunk_overlap  or _settings.chunk_overlap
    splitter = MarkdownTextSplitter(chunk_size=size, chunk_overlap=overlap)

    header_chunks: list[Document] = []
    for doc in docs:
        sections = _MD_HEADER_SPLITTER.split_text(doc.page_content)
        source_path = doc.metadata.get("source", "")
        for section in sections:
            section.metadata["source"] = source_path
        header_chunks.extend(sections)

    final_chunks = splitter.split_documents(header_chunks)

    for chunk in final_chunks:
        source_path = chunk.metadata.get("source", "")
        filename    = Path(source_path).stem if source_path else "未知文件"
        parts = [
            chunk.metadata.get("H1", ""),
            chunk.metadata.get("H2", ""),
            chunk.metadata.get("H3", ""),
            chunk.metadata.get("H4", ""),
        ]
        parts = [p for p in parts if p]
        chunk.metadata["source_name"] = (
            f"{filename} > {' > '.join(parts)}" if parts else filename
        )

    logger.info("md.split.done", files=len(docs), chunks=len(final_chunks))
    return final_chunks


# ── Word 分块 ─────────────────────────────────────────────────

def _is_heading(line: str) -> bool:
    """
    启发式判断一行是否为 Word 标题。

    命中任一条件即视为标题：
      - "第X章/节/篇"、"1."、"1.1"、"一、" 等序号开头的短行
      - 以非句末标点结尾的短行（≤30 字符）
    """
    if not line or len(line) > 30:
        return False
    import re
    if re.match(
        r"^(第[一-龥\d]+[章节篇部分]"
        r"|[一二三四五六七八九十]+[、\.\s]"
        r"|\d+(\.\d+)*[、\.\s]"
        r"|\b[A-Za-z]+\b[:：])",
        line,
    ):
        return True
    # 短行且不以句末标点结尾 → 视为标题
    return line[-1] not in "。！？；，、："


def split_docx_documents(
    docs: list[Document],
    chunk_size: int = 0,          # 0 = 使用配置默认值
    chunk_overlap: int = 0,       # 0 = 使用配置默认值
) -> list[Document]:
    """
    Word 文档分块：按章节标题切分 + 超长章节二次切分。

    Word 经 loader.load_document 加载为纯文本（段落 + 表格行）。
    此函数以启发式 `_is_heading` 识别标题行作为章节边界，逐章节分块；
    单章节文本超过 chunk_size 时，再用 RecursiveCharacterTextSplitter 二次切分。
    块元数据写入 source_name（含章节路径）与 chunk_type。

    Args:
        docs:           loader.load_document() 对 .docx 返回的 list[Document]
        chunk_size:     单块最大字符数；0 表示用配置默认值
        chunk_overlap:  相邻块重叠字符数；0 表示用配置默认值

    Returns:
        list[Document]，metadata 含 source / source_name / chunk_type
    """
    size    = chunk_size    or _settings.chunk_size_docx
    overlap = chunk_overlap or _settings.chunk_overlap
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n", "。", "；", "，", " ", ""],
    )

    final_chunks: list[Document] = []
    for doc in docs:
        source_path = doc.metadata.get("source", "")
        filename    = Path(source_path).stem if source_path else "未知文件"

        lines = [ln.strip() for ln in doc.page_content.split("\n") if ln.strip()]

        # 按标题行切分章节
        sections: list[list[str]] = []          # 每个元素 = 一个章节的行列表
        current: list[str] = []
        for line in lines:
            if _is_heading(line):
                if current:
                    sections.append(current)
                current = [line]
            else:
                current.append(line)
        if current:
            sections.append(current)

        for sec in sections:
            title    = sec[0] if _is_heading(sec[0]) else ""
            sec_text = "\n".join(sec)

            # 短章节整体作一块；超长章节二次切分
            pieces = char_splitter.split_text(sec_text) \
                if len(sec_text) > size else [sec_text]

            for piece in pieces:
                c = Document(page_content=piece, metadata={"source": source_path})
                c.metadata["source_name"] = (
                    f"{filename} > {title}" if title else filename
                )
                c.metadata["chunk_type"] = "text"
                final_chunks.append(c)

    logger.info("docx.split.done", files=len(docs), chunks=len(final_chunks))
    return final_chunks


# ── Excel 分块 ────────────────────────────────────────────────

def split_xlsx_documents(
    docs: list[Document],
    rows_per_chunk: int = 50,
    header_rows: int = 1,
) -> list[Document]:
    """
    Excel 表格分块：每个工作表按行分组，每组保留表头。

    Excel 经 loader.load_document 加载后，每个工作表为一个 Document，
    page_content 为“每行 = 单元格 | 单元格 …”的文本。此处：
      - 取出前 header_rows 行作为表头，始终拼接到每个块的前部，
        保证检索时能带上列名（否则“30mm”之类的单元格失去语义）。
      - 表头之后的数据行每 rows_per_chunk 行为一组生成一个块。
      - 块标记 chunk_type="table"，便于 Milvus 侧按类型过滤或精排加权。

    Args:
        docs:           loader.load_document() 对 .xlsx 返回的 list[Document]
        rows_per_chunk: 每组数据行数
        header_rows:    每个工作表开头的表头行数

    Returns:
        list[Document]，metadata 含 source / sheet / source_name / chunk_type
    """
    final_chunks: list[Document] = []
    for doc in docs:
        source_path = doc.metadata.get("source", "")
        sheet_name  = doc.metadata.get("sheet", "未知工作表")
        filename    = Path(source_path).stem if source_path else "未知文件"

        lines = [ln for ln in doc.page_content.split("\n") if ln.strip()]
        if not lines:
            continue

        header    = lines[:header_rows]
        data_rows = lines[header_rows:]
        header_txt = "\n".join(header)

        if not data_rows:
            # 仅有表头（或单行）的工作表，直接作一个块
            c = Document(
                page_content=header_txt,
                metadata={
                    "source": source_path,
                    "sheet":  sheet_name,
                },
            )
            c.metadata["source_name"] = f"{filename} > {sheet_name}"
            c.metadata["chunk_type"] = "table"
            final_chunks.append(c)
            continue

        group_size = max(1, rows_per_chunk)
        for i in range(0, len(data_rows), group_size):
            group = data_rows[i:i + group_size]
            text  = "\n".join([header_txt, *group])

            row_start = i + 1 + header_rows   # 1-based，含表头偏移
            row_end   = row_start + len(group) - 1

            c = Document(
                page_content=text,
                metadata={"source": source_path, "sheet": sheet_name},
            )
            c.metadata["source_name"] = (
                f"{filename} > {sheet_name} > 第{row_start}~{row_end}行"
            )
            c.metadata["chunk_type"] = "table"
            final_chunks.append(c)

    logger.info("xlsx.split.done", sheets=len(docs), chunks=len(final_chunks))
    return final_chunks


# ── 统一分块入口 ──────────────────────────────────────────────

def split_documents(docs: list[Document], file_path: str) -> list[Document]:
    """统一分块入口，根据文件类型自动选择分块策略"""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return split_pdf_documents(docs)
    elif ext == ".docx":
        return split_docx_documents(docs)
    elif ext == ".xlsx":
        return split_xlsx_documents(docs)
    elif ext in (".md", ".markdown"):
        return split_markdown_documents(docs)
    else:
        raise ValueError(f"不支持的文件类型：{ext}")
