"""
evaluate.py - 诊断树匹配层离线评估（零 LLM / 零 Milvus 依赖）

评估思路：
    data/diag_tree_full.yaml 中 D101+ 节点由 scripts/convert_examples_to_tree.py
    从 data/diag_examples.json（234 条维修实例）确定性转换而来
    （node_id = D{101 + 实例下标}，现象过短/重复的实例被跳过）。

    据此可还原每条实例的「标准答案节点」，再用生产链路同款提取器
    （backend.agents.diagnosis.nodes._extract_phenomena）+ DiagTree.fuzzy_match
    跑匹配，统计 top-1 / top-3 命中率——用于回归验证
    「转换 → 提取 → 匹配」整条离线链路的一致性（改词表/阈值/停用词后必跑）。

指标说明：
    mapped          有标准答案节点的实例数（其余为转换时被跳过：现象过短/重复）
    empty_extract   现象提取结果为空的比例（这些实例只能落 LLM 兜底轨）
    top1 / top3     宽松排名（threshold=0.01，即至少命中 1 个现象词）下，
                    标准节点排第 1 / 前 3 的比例——衡量检索排序质量
    any_hit         严格阈值（默认 0.6，即生产语义）下标准节点进入候选的比例
                    ——源文本含 OCR 乱码碎片，会稀释覆盖率，此指标天然偏低
    code_coverage   文本中提到的故障码能被 exact_match 命中的比例

用法：
    python scripts/evaluate.py [--threshold 0.6] [--show-misses 10]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.agents.diagnosis.diag_tree import DiagTree  # noqa: E402
from backend.agents.diagnosis.nodes import _extract_phenomena  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
EXAMPLES_PATH = BASE / "data" / "diag_examples.json"
TREE_PATH = BASE / "data" / "diag_tree_full.yaml"

# convert_examples_to_tree.py：D101 起，node_id = D{101 + 实例下标}
NODE_START = 101
_FAULT_CODE_RE = re.compile(r"[A-Z]{1,4}[-]?\d{2,4}")


def load_expected_mapping(tree: DiagTree) -> dict[int, str]:
    """还原 实例下标 → 自动转换节点 node_id 的映射（仅保留树中存在的）。"""
    mapping: dict[int, str] = {}
    for idx in range(len(json.loads(EXAMPLES_PATH.read_text(encoding="utf-8")))):
        node_id = f"D{idx + NODE_START}"
        if tree.get(node_id) is not None:
            mapping[idx] = node_id
    return mapping


def evaluate(threshold: float, show_misses: int) -> int:
    if not EXAMPLES_PATH.exists() or not TREE_PATH.exists():
        print(f"缺少数据文件：{EXAMPLES_PATH} / {TREE_PATH}")
        return 1

    tree = DiagTree()
    tree.load_yaml(str(TREE_PATH))
    examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    expected_map = load_expected_mapping(tree)

    n_total = len(examples)
    n_mapped = len(expected_map)
    n_empty = 0
    n_top1 = 0          # 宽松排名（≥1 命中）下的 top-1
    n_top3 = 0
    n_top10 = 0
    n_any_strict = 0    # 严格阈值下进入候选
    misses: list[tuple[int, str, list[str], list[tuple[str, str, float]]]] = []

    for idx, node_id in sorted(expected_map.items()):
        phenomenon = examples[idx].get("phenomenon", "")
        phenomena = _extract_phenomena(phenomenon)
        if not phenomena:
            n_empty += 1
            misses.append((idx, node_id, [], []))
            continue

        # 宽松排名：threshold=0.01 只要求 ≥1 个现象词命中，抗噪声碎片稀释
        loose = [(n.node_id, n.name, round(s, 3))
                 for n, s in tree.fuzzy_match(phenomena, threshold=0.01)]
        strict = [(n.node_id, n.name, round(s, 3))
                  for n, s in tree.fuzzy_match(phenomena, threshold=threshold)]
        rank = next((r for r, m in enumerate(loose, 1) if m[0] == node_id), None)

        if rank == 1:
            n_top1 += 1
        if rank is not None and rank <= 3:
            n_top3 += 1
        if rank is not None and rank <= 10:
            n_top10 += 1
        if any(m[0] == node_id for m in strict):
            n_any_strict += 1
        if rank is None and len(misses) < 500:
            misses.append((idx, node_id, phenomena, loose[:3]))

    # 故障码覆盖：实例文本中的故障码能被精确匹配轨命中
    n_code = 0
    n_code_hit = 0
    for e in examples:
        text = (e.get("phenomenon", "") or "") + (e.get("analysis", "") or "")
        codes = {c.replace("-", "").upper() for c in _FAULT_CODE_RE.findall(text)}
        codes = {c for c in codes if c.startswith("E")}
        if not codes:
            continue
        n_code += 1
        if any(tree.exact_match(c) is not None for c in codes):
            n_code_hit += 1

    # ── 输出报告 ──────────────────────────────────────────────
    line = "=" * 56
    print(line)
    print(" 诊断树匹配层离线评估（examples → 转换节点 roundtrip）")
    print(line)
    print(f" 诊断树节点数        : {len(tree.get_tree()['nodes'])}")
    print(f" 评估实例总数        : {n_total}")
    print(f" 有标准答案(mapped)  : {n_mapped}  （其余 {n_total - n_mapped} 条转换时被跳过）")
    denom = n_mapped or 1
    print(f" 现象提取为空        : {n_empty}  ({n_empty / denom:.1%})")
    print(f" top-1 命中(宽松)    : {n_top1}  ({n_top1 / denom:.1%})")
    print(f" top-3 命中(宽松)    : {n_top3}  ({n_top3 / denom:.1%})")
    print(f" top-10 命中(宽松)   : {n_top10}  ({n_top10 / denom:.1%})")
    print(f" 进入候选(严格阈值)  : {n_any_strict}  ({n_any_strict / denom:.1%})")
    print(f" 严格阈值            : fuzzy_match threshold = {threshold}")
    print(f" 故障码覆盖率        : {n_code_hit}/{n_code}  "
          f"({(n_code_hit / n_code):.1%})" if n_code else " 故障码覆盖率        : 无含故障码实例")
    print(line)

    if show_misses and misses:
        print(f"\n未命中样例（最多 {show_misses} 条）:")
        for idx, node_id, phenomena, ranked in misses[:show_misses]:
            print(f"  [{idx}] 期望 {node_id}  提取={phenomena}")
            if ranked:
                print(f"        候选: {ranked}")
            else:
                print("        候选: 无（低于阈值）")
        if len(misses) > show_misses:
            print(f"  ... 其余 {len(misses) - show_misses} 条未展示")

    return 0


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="诊断树匹配层离线评估")
    parser.add_argument("--threshold", type=float, default=0.6, help="fuzzy_match 阈值（默认 0.6）")
    parser.add_argument("--show-misses", type=int, default=10, help="展示未命中样例条数（默认 10）")
    args = parser.parse_args()
    sys.exit(evaluate(args.threshold, args.show_misses))
