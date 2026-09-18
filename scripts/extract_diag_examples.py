"""
extract_diag_examples.py - 从数控机床维修手册 PDF 提取故障诊断实例

数据来源：最新数控机床加工工艺编程技术与维护维修实用手册
（06第六编数控机床第07七编数控机床维修实例.pdf，第七编 202-448 页）

每个实例输出结构：
    {
        "source_page": 页码,
        "title":       实例标题（若可识别）,
        "phenomenon":  故障现象,
        "analysis":    故障检查与分析,
        "treatment":   故障处理 / 排除方法,
    }

产物：data/diag_examples.json（供 ⑤号 挑选扩充 diag_tree.yaml 或知识库种子数据）

用法：
    python scripts/extract_diag_examples.py <pdf路径> [输出路径]
"""
import json
import re
import sys
from pathlib import Path

import fitz

DEFAULT_PDF = (
    r"D:\BaiduNetdiskDownload\最新数控机床加工工艺编程技术与维护维修实用手册"
    r"\最新数控机床加工工艺编程技术与维护维修实用手册"
    r"\最新数控机床加工工艺编程技术与维护维修实用手册"
    r"\最新数控机床加工工艺编程技术与维护维修实用手册"
    r"\06第六编数控机床第07七编数控机床维修实例.pdf"
)
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "diag_examples.json"

# 第七编起始页（含"数控机床维修实例"）
PART7_START = 202


def clean_text(s: str) -> str:
    """去空白/页码污染，保留可读内容。"""
    s = re.sub(r"[ \t\u3000]+", "", s)          # 去空格
    s = re.sub(r"·\s*[\d+\-]+\s*·", "", s)      # 去页码装饰（·123·）
    s = re.sub(r"最新数控机床加工工艺编程技术与维护维修实用手册", "", s)
    s = re.sub(r"第六编\s*数控机床故障诊断及维护维修", "", s)
    s = re.sub(r"第七编\s*数控机床维修实例", "", s)
    return s.strip()


def extract_examples(pdf_path: str) -> list[dict]:
    """按『故障现象』锚点提取实例（现象/检查分析/处理三段）。"""
    doc = fitz.open(pdf_path)
    text = "\n".join(doc[i].get_text() for i in range(PART7_START - 1, doc.page_count))

    results = []
    anchors = list(re.finditer(r"故障现象[：:]", text))
    for m in anchors:
        pos = m.start()
        # 现象段：锚点 → 下一个"故障检查/分析/处理"
        rest = text[pos + 4: pos + 500]
        end = re.search(r"故障(?:检查|分析)[：:]", rest)
        phenomenon = clean_text(rest[:end.start()] if end else rest[:120])[:120]
        # 现象里可能混入"故障检查与分析："前缀，剥掉
        phenomenon = re.sub(r"故障(?:检查|分析)[：:].*$", "", phenomenon)

        # 检查/分析段：锚点 → 故障处理
        analysis = ""
        if end:
            seg = text[pos + 4 + end.start(): pos + 2500]
            cut = re.search(r"故障处理[：:]", seg)
            analysis = clean_text(seg[:cut.start()] if cut else seg[:400])[:400]

        # 处理段：故障处理 → 下一个实例锚点/【例
        treatment = ""
        tm = re.search(r"故障处理[：:]\s*(.{0,1200}?)(?=【例|故障现象[：:]|\Z)", text[pos:pos + 5000], re.S)
        if tm:
            treatment = clean_text(tm.group(1))[:300]

        if phenomenon and treatment:
            results.append({
                "phenomenon": phenomenon,
                "analysis": analysis,
                "treatment": treatment,
            })

    # 去掉明显错位（现象为空 / 处理重复引用）
    dedup = []
    seen = set()
    for r in results:
        key = r["phenomenon"][:20]
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return dedup


def main() -> None:
    pdf = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT

    examples = extract_examples(pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    full = [e for e in examples if e["phenomenon"] and e["treatment"]]
    print(f"提取实例总数: {len(examples)}")
    print(f"现象+处理完整: {len(full)}")
    print(f"输出: {out}")
    for e in full[:10]:
        print(f"  · {e['phenomenon'][:44]}")


if __name__ == "__main__":
    main()
