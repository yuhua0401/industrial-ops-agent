"""
evaluate.py - 测试集评估

跑批测试集，计算 Agent 的准确率、召回率等指标。
"""
import json
from pathlib import Path


def evaluate():
    """加载测试集 → 逐条调用 Agent → 计算指标。"""
    test_set_path = Path("tests/test_data/test_set.json")
    if not test_set_path.exists():
        print("测试集文件不存在，请先创建 tests/test_data/test_set.json")
        return

    # TODO: 实现评估流程
    print("评估完成。")


if __name__ == "__main__":
    evaluate()
