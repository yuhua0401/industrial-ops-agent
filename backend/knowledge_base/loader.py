"""
loader - 文档加载器

支持 PDF、Word、Markdown、Excel 等格式的文档解析。
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from backend.core.logger import get_logger

logger = get_logger(__name__)


def load_pdf(file_path: str) -> list[Document]:
    """
    加载 PDF 文档，每页返回一个 Document。

    只提取文字层内容；图片/扫描件页面 page_content 为空字符串，
    不报错（在分块时 splitter.py 会过滤掉空页）。

    Args:
        file_path: PDF 文件的本地路径

    Returns:
        list[Document]，每个 Document 对应一页
        metadata 包含 source（文件路径）和 page（页码，从 0 开始）
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    logger.info("load.pdf.done", pages=len(pages), file=Path(file_path).name)
    return pages


def load_markdown(file_path: str) -> list[Document]:
    """
    加载 Markdown 文档，整个文件作为一个 Document 返回。

    不在这里做标题切分——那是分块步骤（splitter.py）的工作。
    这里只负责把文件内容读进内存。

    Args:
        file_path: Markdown 文件的本地路径（.md 或 .markdown）

    Returns:
        list[Document]，只有一个元素，page_content 为文件全文
        metadata 包含 source（文件路径）
    """
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    char_count = len(docs[0].page_content)
    logger.info("load.md.done", chars=char_count, file=Path(file_path).name)
    return docs


def load_docx(file_path: str) -> list[Document]:
    """
    加载 Word 文档，整个文件作为一个 Document 返回。

    同时提取段落文本和表格内容（表格按行以 '|' 拼接），
    再按空行去重后的段落顺序拼接为全文。

    Args:
        file_path: Word 文件的本地路径（.docx）

    Returns:
        list[Document]，只有一个元素，page_content 为全文
        metadata 包含 source（文件路径）
    """
    from docx import Document as DocxDocument

    doc = DocxDocument(file_path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    text = "\n".join(parts)
    logger.info("load.docx.done", chars=len(text), file=Path(file_path).name)
    return [Document(page_content=text, metadata={"source": file_path})]


def load_xlsx(file_path: str) -> list[Document]:
    """
    加载 Excel 文档，每个工作表作为一个 Document 返回。

    读取所有工作表，将每个工作表的单元格按行拼接为文本，
    保留表头，单元格之间以 ' | ' 分隔。

    Args:
        file_path: Excel 文件的本地路径（.xlsx）

    Returns:
        list[Document]，每个元素对应一个工作表
        metadata 包含 source（文件路径）和 sheet（工作表名）
    """
    from openpyxl import load_workbook

    wb = load_workbook(file_path, read_only=True, data_only=True)
    docs: list[Document] = []
    for sheet in wb.worksheets:
        rows: list[str] = []
        for row in sheet.iter_rows(values_only=True):
            cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
            if cells:
                rows.append(" | ".join(cells))
        text = "\n".join(rows)
        docs.append(
            Document(page_content=text, metadata={"source": file_path, "sheet": sheet.title})
        )
    wb.close()
    logger.info("load.xlsx.done", sheets=len(docs), file=Path(file_path).name)
    return docs


def load_document(file_path: str) -> list[Document]:
    """
    统一文档加载入口。

    根据文件扩展名自动选择加载器：
        .pdf            → PyPDFLoader（文字层提取）
        .docx           → python-docx（段落 + 表格文本）
        .xlsx           → openpyxl（每个工作表一个 Document）
        .md / .markdown → TextLoader（纯文本读取）

    Args:
        file_path: 文档本地路径

    Returns:
        list[Document]

    Raises:
        ValueError: 不支持的文件类型
        FileNotFoundError: 文件不存在
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext == ".xlsx":
        return load_xlsx(file_path)
    elif ext in (".md", ".markdown"):
        return load_markdown(file_path)
    else:
        raise ValueError(
            f"不支持的文件类型：{ext}\n"
            f"当前支持：.pdf / .docx / .xlsx / .md / .markdown"
        )
