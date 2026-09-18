"""
diag_tree - 故障诊断树数据结构

融合说明：
- 项目 DiagnosisTreeNode（Pydantic 模型，供数据维护/导入导出）；
-  DiagTree（YAML 加载 / 故障码精确匹配 / 现象 Jaccard 模糊匹配 / get_tree），
"""
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field


class DiagnosisTreeNode(BaseModel):
    """诊断树节点（项目原有，Pydantic 版）。"""
    device_model: str = Field(description="适用设备型号")
    fault_category: str = Field(description="故障类别，如'电源异常'、'通讯故障'")
    symptoms: list[str] = Field(description="故障现象关键词列表")
    possible_causes: list[str] = Field(description="可能原因")
    steps: list[str] = Field(description="排查步骤，按顺序执行")
    solution: str = Field(default="", description="解决方案")
    severity: str = Field(default="medium")
    requires_engineer: bool = Field(default=False, description="是否需要工程师上门")


@dataclass
class RootCause:
    desc: str
    probability: str = "中"      # 高/中/低


@dataclass
class Solution:
    step: str
    need_skill: bool = False     # False=用户可自行处理


@dataclass
class DiagNode:
    node_id:    str
    name:       str
    fault_code: str | None = None
    phenomena:  list[str] = field(default_factory=list)
    priority:   int = 1                    # 1低 2中 3高
    causes:     list[RootCause] = field(default_factory=list)
    solutions:  list[Solution] = field(default_factory=list)
    need_ticket: bool = False
    questions:  list[str] = field(default_factory=list)   # 追问问题
    children:   list[str] = field(default_factory=list)   # 子节点 id

    def to_dict(self) -> dict:
        return {
            "node_id":    self.node_id,
            "fault_code": self.fault_code,
            "name":       self.name,
            "priority":   self.priority,
            "phenomena":  self.phenomena,
            "causes":     [{"desc": c.desc, "probability": c.probability} for c in self.causes],
            "solutions":  [{"step": s.step, "need_skill": s.need_skill} for s in self.solutions],
            "need_ticket": self.need_ticket,
            "questions":  self.questions,
        }


class DiagTree:
    """诊断树：支持故障码精确匹配 + 现象模糊匹配（Jaccard 相似度）。"""

    def __init__(self) -> None:
        self._nodes: dict[str, DiagNode] = {}
        self._code_index: dict[str, str] = {}     # fault_code -> node_id

    # ── 加载 ────────────────────────────────────────────────
    def load_yaml(self, path: str) -> None:
        """从 YAML 数据文件加载诊断树（数据文件由 ⑤号 维护）。"""
        import yaml
        with open(path, encoding="utf-8") as f:
            raw_nodes = yaml.safe_load(f) or []
        for raw in raw_nodes:
            node = DiagNode(
                node_id    = raw["node_id"],
                name       = raw.get("name", ""),
                fault_code = raw.get("fault_code"),
                phenomena  = raw.get("phenomena", []),
                priority   = int(raw.get("priority", 1)),
                causes     = [RootCause(**c) for c in raw.get("causes", [])],
                solutions  = [Solution(**s) for s in raw.get("solutions", [])],
                need_ticket= bool(raw.get("need_ticket", False)),
                questions  = raw.get("questions", []),
                children   = raw.get("children", []),
            )
            self.add_node(node)

    def add_node(self, node: DiagNode) -> None:
        self._nodes[node.node_id] = node
        if node.fault_code:
            self._code_index[node.fault_code.upper()] = node.node_id

    # ── 匹配 ────────────────────────────────────────────────
    def exact_match(self, code: str | None) -> DiagNode | None:
        """故障码精确匹配。code 为 None 或未命中返回 None。"""
        if not code:
            return None
        node_id = self._code_index.get(code.strip().upper())
        return self._nodes.get(node_id)

    # 通用/泛化现象词：单凭这些词不足以区分故障（大量自动转换节点都有），
    # 匹配时剔除，避免"电机不转+过载报警"误命中 100+ 个仅含"报警"的节点。
    _GENERIC_PHENOMENA = frozenset({
        "报警", "不转", "停止", "故障", "不准", "不动", "异常", "抖动",
        "振动", "显示", "系统", "超程", "失灵", "错位", "噪声", "异响",
        "失效", "松动", "失控", "断线", "不稳定", "不动作", "不进给",
    })

    def fuzzy_match(self, phenomena: list[str], threshold: float = 0.6) -> list[tuple[DiagNode, float]]:
        """
        现象模糊匹配：覆盖率 = 命中的输入现象数 / 输入现象总数。

        命中判定支持子串包含（双向）：
            输入 "电机不转了" 命中节点现象 "电机不转"（输入 ⊇ 节点）
            输入 "通讯"      命中节点现象 "通讯中断"（输入 ⊆ 节点）
        配合 nodes._extract_phenomena 的词缀剥离（了/过/在），可大幅提升口语化
        描述的命中率，避免全部落到 LLM 兜底轨。

        通用现象词（_GENERIC_PHENOMENA）不参与计数：输入和节点中的此类词
        都会被忽略，防止低区分度词造成批量误命中。

        返回 [(node, score), ...] 按 score 降序，低于 threshold 的剔除。
        """
        if not phenomena:
            return []
        input_set = {
            p.strip() for p in phenomena if p.strip()
            and p.strip() not in self._GENERIC_PHENOMENA
        }
        if not input_set:
            return []
        scored = []
        for node in self._nodes.values():
            node_set = set(node.phenomena) - self._GENERIC_PHENOMENA
            if not node_set:
                continue
            hit = sum(
                1 for p in input_set
                if any(p in np or np in p for np in node_set)
            )
            score = hit / len(input_set)
            if score >= threshold:
                scored.append((node, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def get(self, node_id: str) -> DiagNode | None:
        return self._nodes.get(node_id)

    def get_tree(self) -> dict:
        """获取诊断树全量结构（供 api/admin 展示）。"""
        return {"nodes": [n.to_dict() for n in self._nodes.values()]}


# ──────────────────────────────────────────────────────────────
# 直接运行演示：python diag_tree.py
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data = Path(__file__).resolve().parents[3] / "data" / "diag_tree_full.yaml"
    print(f"诊断树数据文件: {data}")
    tree = DiagTree()
    if not data.exists():
        print("诊断树数据文件不存在")
        raise SystemExit(1)

    tree.load_yaml(str(data))
    nodes_list = tree.get_tree()["nodes"]
    print(f"加载 {len(nodes_list)} 个节点:")
    for n in nodes_list:
        print(f"  {n['node_id']} {n['name']} fault_code={n.get('fault_code')} priority={n.get('priority')}")

    print("\n=== 故障码精确匹配（大小写不敏感）===")
    for code in ["e001", "E002", "E999", None]:
        node = tree.exact_match(code)
        print(f"  exact_match({code!r}) → {node.node_id + ' ' + node.name if node else None}")

    print("\n=== 现象模糊匹配（覆盖率降序，低于阈值剔除）===")
    for phenomena, threshold in [(["电机不转", "过载报警"], 0.5), (["通讯中断"], 0.5)]:
        matches = tree.fuzzy_match(phenomena, threshold)
        print(f"  fuzzy_match({phenomena}, {threshold}) → {[(n.node_id, n.name, round(s, 2)) for n, s in matches]}")
