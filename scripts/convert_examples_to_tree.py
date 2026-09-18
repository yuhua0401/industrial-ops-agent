"""
convert_examples_to_tree.py - 将维修实例批量转换为诊断树节点

输入：data/diag_examples.json（extract_diag_examples.py 产物，234 条）
输出：data/diag_tree_full.yaml（追加到现有 diag_tree.yaml 之后的完整诊断树）

转换规则：
- node_id: D101 起（避开已有 D001-D013）
- name: 从现象中提取 8-14 字主题
- phenomena: 现象清洗后的关键词列表（2-8 字片段）
- causes: 从 analysis/treatment 提取根因句（"由于…/…所致/故障为…"等模式）
- solutions: 从 treatment 按序号/分号切分为步骤
- need_ticket: 含"更换/专业人员/厂家"等高风险词时为 true
- questions: 从现象中提取 1 个追问

用法：python scripts/convert_examples_to_tree.py
"""
import json
import re
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "data" / "diag_examples.json"
OUT = BASE / "data" / "diag_tree_full.yaml"
# 已有节点 D001-D013，新节点从 D101 开始
NODE_START = 101

# 清洗用：去掉字体嵌入产生的乱码（拉丁字母/数字被错误映射），保留中文与常用标点
GARBAGE_RE = re.compile(r"[#$%&'()*+,./:;<=>?@\\^_`|~]")
# 现象里混入的"故障检查与分析"尾巴
TAIL_RE = re.compile(r"故障(?:检查|分析)[：:].*$")
# 根因句模式
CAUSE_PATTERNS = [
    r"(?:由于|因|是因为)[^，。；；]{4,40}",
    r"[^，。；；]{2,40}(?:所致|引起|造成|导致)[^，。；；]{0,15}",
    r"故障(?:为|是|点)[^，。；；]{2,30}",
    r"(?:原因|问题)(?:是|为|在)[^，。；；]{2,30}",
]
# 处理步骤分隔：序号（1. 2. ① ② ! " #）或分号/句号
STEP_SPLIT_RE = re.compile(r"[；;。]|(?:\d+[、.．])|(?:[①②③④⑤⑥⑦⑧⑨⑩])")
TICKET_WORDS = ["更换", "专业人员", "厂家", "上门", "维修中心", "拆机"]


def clean(s: str) -> str:
    s = GARBAGE_RE.sub("", s)
    s = TAIL_RE.sub("", s)
    s = re.sub(r"^[：:\s]+", "", s)
    s = re.sub(r"\s+", "", s)
    return s.strip()


def extract_name(phenomenon: str, phenomena: list[str]) -> str:
    """
    从现象中提取主题名：
    优先用「部件+故障」原子词（如 '主轴不转'）；其次用最长中文片段。
    """
    if phenomena:
        # 优先取含部件词的组合（如 "主轴不转"），否则取第一个原子词
        for p in phenomena:
            if any(part in p for part in PART_WORDS):
                return p[:14]
        return phenomena[0][:14]
    cn = chinese_only(phenomenon)
    if not cn:
        return "设备故障排查"
    segs = re.findall(r"[\u4e00-\u9fff]{4,}", phenomenon)
    best = max(segs, key=len) if segs else cn
    return best[:12]


def chinese_only(s: str) -> str:
    """只保留连续汉字（用于现象关键词/名称，过滤字体乱码）。"""
    return "".join(re.findall(r"[\u4e00-\u9fff]+", s))


# 部件词表（按最长优先匹配）
PART_WORDS = [
    "光栅尺", "滚珠丝杠", "机械手", "编码器", "继电器", "伺服单元", "电动机",
    "步进电动机", "主轴", "刀架", "刀库", "工作台", "显示屏", "丝杠",
    "导轨", "换刀", "卡盘", "润滑", "电源", "系统", "刀塔", "冷却",
]
# 故障表现词表
FAULT_WORDS = [
    "乱字符", "不能起动", "不能移动", "不能控制", "不能运行", "不能输入",
    "不到位", "越位", "失步", "超差", "失灵", "抖动", "振动", "报警",
    "烧坏", "断线", "失控", "无显示", "噪声", "错位", "超程", "卡死",
    "松动", "漏油", "掉电", "停止", "失效", "异响", "不转", "不准",
    "不稳定", "不动作", "时转时不转", "不进给",
    "发热",
    "不显示",
    "转不动",
    "不能转",
    "不动作",
    "不工作",
    "失控",
    "超程",
    "爬行",
    "窜动",
    "跳动",
    "摆动",
    "堵转",
    "闷车",
    "闪断",
    "误动作",
    "不换刀",
    "卡刀",
    "掉刀",
    "撞刀",
    "过切",
    "欠切",
    "偏心",
]


def extract_phenomena(phenomenon: str, name: str) -> list[str]:
    """
    从现象中提取原子故障关键词（部件词+故障表现词）。
    短原子词才能被 fuzzy_match 的双向子串判定命中，长句碎片无法匹配。
    """
    p = clean(phenomenon)
    if not p:
        return [name[:6]]
    # 剥掉"故障检查与分析"尾巴
    p = re.split(r"故障(?:检查|分析)", p)[0]
    # 去掉标点/空白，只留中文连续串
    p = "".join(re.findall(r"[\u4e00-\u9fff]+", p))
    if not p:
        return [name[:6]]

    segs = []
    # 1) 部件词 + 后续故障词 组合（如 "主轴"+"不转" → "主轴不转"）
    for part in PART_WORDS:
        idx = p.find(part)
        if idx == -1:
            continue
        tail = p[idx + len(part): idx + len(part) + 12]
        for fault in FAULT_WORDS:
            if fault in tail:
                segs.append(part + fault)
                break
    # 2) 独立故障词（无部件限定）
    for fault in FAULT_WORDS:
        if fault in p:
            segs.append(fault)
    # 3) 兜底：最长中文片段前 6 字
    if not segs:
        segs = [max(re.findall(r"[\u4e00-\u9fff]{2,}", p), key=len, default=name)[:6]]
    # 去重保序，最多 4 个
    seen, out = set(), []
    for s in segs:
        if s not in seen and 2 <= len(s) <= 10:
            seen.add(s)
            out.append(s)
    return out[:4]


def extract_causes(analysis: str, treatment: str) -> list[dict]:
    """从分析/处理中提取根因句。"""
    src = analysis or treatment
    if not src:
        return [{"desc": "需现场排查确认", "probability": "中"}]
    found = []
    for pat in CAUSE_PATTERNS:
        for m in re.finditer(pat, src):
            s = clean(m.group(0))
            if 4 <= len(s) <= 40 and s not in found:
                found.append(s)
            if len(found) >= 3:
                break
        if len(found) >= 3:
            break
    if not found:
        # 从处理动词反推
        verb = re.search(r"(更换|修复|调整|清洗|重新装配|紧固)\S{0,10}", treatment)
        if verb:
            found.append(f"{verb.group(0)}相关部件异常")
    if not found:
        return [{"desc": "需现场排查确认", "probability": "中"}]
    probs = ["高", "中", "低"]
    return [{"desc": d, "probability": probs[i] if i < 3 else "低"}
            for i, d in enumerate(found[:3])]


def extract_solutions(treatment: str) -> list[dict]:
    """从处理文本切分为步骤。"""
    t = clean(treatment)
    if not t:
        return [{"step": "建议联系专业维修人员处理", "need_skill": True}]
    parts = [s for s in STEP_SPLIT_RE.split(t) if 4 <= len(s) <= 60]
    if not parts:
        parts = [t[:50]]
    steps = []
    for s in parts[:4]:
        need_skill = any(w in s for w in TICKET_WORDS) or len(s) > 25
        steps.append({"step": s, "need_skill": need_skill})
    return steps


def extract_question(phenomenon: str, name: str) -> str:
    """从现象生成 1 个追问问题。"""
    key = re.sub(r"[，。！？、；：,.!?;:：]", "", clean(phenomenon))[:20]
    return f"请描述「{name}」的具体表现（{key}）？"


def convert() -> None:
    with open(SRC, encoding="utf-8") as f:
        examples = json.load(f)

    nodes = []
    seen = set()
    for idx, e in enumerate(examples):
        phenomenon = e.get("phenomenon", "")
        analysis = e.get("analysis", "")
        treatment = e.get("treatment", "")

        # 清洗
        ph_clean = clean(phenomenon)
        if len(ph_clean) < 6:
            continue  # 现象太短，跳过
        # 去重（按现象前 12 字）
        key = ph_clean[:12]
        if key in seen:
            continue
        seen.add(key)

        ph_words = extract_phenomena(ph_clean, "")
        name = extract_name(ph_clean, ph_words)
        nodes.append({
            "node_id": f"D{idx + NODE_START}",
            "name": name,
            "phenomena": ph_words,
            "causes": extract_causes(analysis, treatment),
            "solutions": extract_solutions(treatment),
            "need_ticket": any(w in treatment for w in TICKET_WORDS),
            "questions": [extract_question(ph_clean, name)],
        })

    # 合并现有诊断树 + 新节点
    # 注：早期版本引用 backend/agents/diagnosis/data/diag_tree.yaml（该路径不存在），
    # 合并分支从不生效；现修正为项目根的 data/diag_tree_full.yaml。
    existing = []
    src_tree = BASE / "data" / "diag_tree_full.yaml"
    if src_tree.exists():
        with open(src_tree, encoding="utf-8") as f:
            existing = yaml.safe_load(f) or []

    all_nodes = existing + nodes
    with open(OUT, "w", encoding="utf-8") as f:
        yaml.safe_dump(all_nodes, f, allow_unicode=True, sort_keys=False)

    print(f"输入实例: {len(examples)}")
    print(f"跳过(现象过短/重复): {len(examples) - len(nodes)}")
    print(f"新增节点: {len(nodes)}")
    print(f"合并后总数: {len(all_nodes)} (D001-D013 已有 + 新增)")
    print(f"输出: {OUT}")


if __name__ == "__main__":
    convert()
