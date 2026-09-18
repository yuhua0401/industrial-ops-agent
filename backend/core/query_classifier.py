"""
query_classifier - MiniLM 意图分类

Author:hiema1
Date:2026/8/10
Version:0.0.1
"""


# backend/core/query_classifier.py

import json
import os
import random
from pathlib import Path
from typing import Any, Optional

import torch

from backend.config import get_settings
from backend.core.logger import get_logger

backend_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

logger = get_logger(__name__)

LABEL2ID = {"general": 0, "specialized": 1}
ID2LABEL  = {0: "general",  1: "specialized"}

# general 侧置信度阈值设为 0.85（偏高）：
# 专业问题被误判为通用问题的代价更高——LLM 会用自身知识回答，
# 可能与设备手册内容矛盾。宁可多走一次 RAG，不要漏掉设备相关问题。
GENERAL_CONFIDENCE_THRESHOLD = 0.85

class QueryClassifier:
    """
    QA Query 二分类器：general / specialized。

    训练阶段：
        qc = QueryClassifier()
        qc.train("backend/training_data.jsonl", output_dir="models/classifier")

    推理阶段：
        qc = QueryClassifier("models/classifier")
        label, conf = qc.classify("什么是 Spring IOC？")
    """

    _instance: Optional["QueryClassifier"] = None
    def __init__(self, model_path: str | None = None):
        """
        Args:
            model_path:
                - 传入路径 → 加载该路径的模型（可用于加载基座或任意微调模型）
                - None（默认）→ 加载训练好的微调模型（FINETUNED_CLASSIFIER_PATH）
        """
        settings = get_settings()

        if model_path:
            # 显式传入：用于训练时加载基座，或临时切换其他模型
            model_id = model_path
            base_path = os.path.join(backend_path, settings.classifier_model_path)
            self._is_finetuned = model_path != base_path
        else:
            # 默认：加载微调模型（假设已训练完成）
            model_id = os.path.join(backend_path, settings.finetuned_classifier_path)
            self._is_finetuned = True
        device = 0 if torch.cuda.is_available() else -1
        from transformers import pipeline as hf_pipeline
        self._pipeline = hf_pipeline(task="text-classification",
                                     model=model_id,
                                     device=device,
                                     top_k=None,
                                     truncation=True,
                                     max_length=128,)
        logger.info("query_classifier.loaded", model_id=model_id, finetuned=self._is_finetuned)

    @classmethod
    def get_instance(cls) -> "QueryClassifier":
        """获取单例（首次调用时懒加载）"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── 训练 ─────────────────────────────────────────────────

    def train(
        self,
        data_path: str,
        output_dir: str,
        epochs: int = 8,
        batch_size: int = 64,
        lr: float = 2e-5,
        max_length: int = 128,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        """
        微调当前加载的基座模型，训练完自动保存到 output_dir。

        Args:
            data_path:  训练数据路径（JSONL，每行 {"text": ..., "label": "general/specialized"}）
            output_dir: 微调模型保存目录（训练完可直接用此路径初始化新实例）
            epochs:     训练轮数（默认 8）
            batch_size: 训练批大小（默认 64）
            lr:         学习率（默认 2e-5）
            max_length: Token 最大长度（默认 128，Query 分类用不到长文本）
            val_ratio:  验证集比例（默认 0.1）
            test_ratio: 测试集比例（默认 0.1）
            seed:       随机种子（默认 42）
        """
        # 训练库仅在此方法内 import，推理路径不加载这些依赖
        import numpy as np
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )

        set_seed(seed)

        # ── 加载数据 ──────────────────────────────────────────
        rows = self._load_jsonl(data_path)
        train_rows, val_rows, test_rows = self._stratified_split(
            rows, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
        logger.info("query_classifier.split",
                    train=len(train_rows), val=len(val_rows), test=len(test_rows))

        # ── Tokenizer + Model ─────────────────────────────────
        # 从当前 pipeline 取 model_id，保持一致
        model_id = self._pipeline.model.config._name_or_path
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            ignore_mismatched_sizes=True,   # 分类头从2类随机初始化
        )

        # ── 数据集 ────────────────────────────────────────────
        def to_dataset(rows_: list[dict]) -> Dataset:
            return Dataset.from_dict({
                "text":  [r["text"]              for r in rows_],
                "label": [LABEL2ID[r["label"]]   for r in rows_],
            })

        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True, max_length=max_length)

        train_ds = to_dataset(train_rows).map(tokenize, batched=True, remove_columns=["text"])
        val_ds   = to_dataset(val_rows).map(tokenize,   batched=True, remove_columns=["text"])
        test_ds  = to_dataset(test_rows).map(tokenize,  batched=True, remove_columns=["text"])

        # ── 评估指标 ──────────────────────────────────────────
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, preds)
            _, _, f1, _ = precision_recall_fscore_support(
                labels, preds, average="macro", zero_division=0
            )
            return {"accuracy": float(acc), "f1_macro": float(f1)}

        # ── TrainingArguments ─────────────────────────────────
        use_cuda = torch.cuda.is_available()
        checkpoint_dir = str(Path(output_dir) / "_checkpoints")

        train_args = TrainingArguments(
            output_dir=checkpoint_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,     # 训练结束自动加载最优 checkpoint
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=lr,
            warmup_ratio=0.1,
            weight_decay=0.01,
            save_total_limit=1,
            logging_steps=20,
            fp16=use_cuda,
            report_to="none",
            seed=seed,
        )

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        # ── 训练 + 测试集评估 ──────────────────────────────────
        trainer.train()
        test_metrics = trainer.evaluate(test_ds)
        logger.info("query_classifier.test_metrics", **{
            k: round(v, 4) for k, v in test_metrics.items() if k.startswith("eval_")
        })

        # ── 保存 ──────────────────────────────────────────────
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("query_classifier.saved", output_dir=output_dir)
        print(f"\n✅ 微调完成，模型保存到：{output_dir}")
        print(f"   下次使用：QueryClassifier('{output_dir}')")

    # ── 推理 ─────────────────────────────────────────────────

    def classify(self, text: str) -> tuple[str, float]:
        """
        对 query 做 general / specialized 二分类。

        Returns:
            (label, confidence)
            label:      "general" 或 "specialized"
            confidence: 对应标签的置信度 [0, 1]

        分类规则：
            P(general) >= 0.85 → ("general",      P(general))
            其余               → ("specialized",  1 - P(general))

        兜底：标签名不匹配时保守返回 ("specialized", 0.5)。
        """
        raw_outputs: list[dict] = self._pipeline(text)[0]

        # 查找 general 标签的分数（兼容大小写和 LABEL_0 格式）
        general_score: float | None = None
        for item in raw_outputs:
            lbl = item["label"].lower()
            if lbl in ("general", "label_0"):
                general_score = item["score"]
                break

        if general_score is None:
            logger.warning("query_classifier.unexpected_labels",
                           labels=[x["label"] for x in raw_outputs])
            return "specialized", 0.5

        if general_score >= GENERAL_CONFIDENCE_THRESHOLD:
            label, confidence = "general", general_score
        else:
            label, confidence = "specialized", 1.0 - general_score

        logger.info("query_classifier.result",
                    text_preview=text[:50],
                    label=label,
                    confidence=round(confidence, 4))
        return label, confidence

    # ── 私有工具方法 ─────────────────────────────────────────

    @staticmethod
    def _load_jsonl(path: str) -> list[dict]:
        rows: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text  = (obj.get("text") or "").strip()
                label = obj.get("label")
                if not text:
                    raise ValueError(f"第 {line_no} 行 text 为空")
                if label not in LABEL2ID:
                    raise ValueError(f"第 {line_no} 行 label 非法: {label!r}")
                rows.append({"text": text, "label": label})
        if not rows:
            raise ValueError(f"训练数据为空：{path}")
        return rows

    @staticmethod
    def _stratified_split(
            rows: list[dict],  # 原始数据列表，每个元素是 {"text": ..., "label": "general"/"specialized"}
            val_ratio: float,  # 验证集比例（如 0.1）
            test_ratio: float,  # 测试集比例（如 0.1）
            seed: int,  # 随机种子，保证可复现
    ) -> tuple[list[dict], list[dict], list[dict]]:  # 返回 (训练集, 验证集, 测试集)
        """按标签做分层切分，保证训练/验证/测试集的类别比例一致。"""
        # 设置随机数种子，使得每次运行切分结果相同（便于复现实验）
        random.seed(seed)
        # 创建一个空字典，用于按标签（label）分组存放数据
        # key 为标签名（如 "general"），value 为该标签对应的所有数据列表
        buckets: dict[str, list[dict]] = {}
        # 遍历每一条数据，将其放入对应标签的桶中
        for row in rows:
            # setdefault：如果桶中已有该标签，则返回已有的列表；否则新建一个空列表并返回
            # 然后 .append(row) 把当前数据追加到该列表中
            buckets.setdefault(row["label"], []).append(row)
        # 调试打印
        # print(f'buckets-->{len(buckets)}')
        # print(f'buckets-->{buckets}')
        # 初始化三个列表，分别存放训练集、验证集、测试集的数据
        train_rows, val_rows, test_rows = [], [], []
        # 遍历每个标签组（每个桶），对每个类别单独进行切分，保证各类别在三个集合中比例一致
        for _label, group in buckets.items():
            # 随机打乱当前组内的数据顺序，防止原始顺序带来的偏差
            random.shuffle(group)
            # 获取当前组的总样本数
            n = len(group)
            # 计算测试集应取的数量：至少取 1 条（max(1, ...) 防止小样本时取到 0）
            n_test = max(1, int(n * test_ratio))
            # 计算验证集应取的数量：至少取 1 条
            n_val = max(1, int(n * val_ratio))
            # 从打乱后的组中取前 n_test 条作为测试集
            test_rows.extend(group[:n_test])
            # 接着取接下来的 n_val 条作为验证集
            val_rows.extend(group[n_test:n_test + n_val])
            # 剩余的全部作为训练集
            train_rows.extend(group[n_test + n_val:])
        # 最后将训练集整体再打乱一次，避免训练时类别或样本顺序过于集中
        random.shuffle(train_rows)

        # 返回三个集合
        return train_rows, val_rows, test_rows


# ── 模块级单例（供 nodes.py import 调用）───────────────────────

_classifier: QueryClassifier | None = None


# ── 模块级单例（供 nodes.py import 调用）───────────────────────

def get_query_classifier() -> QueryClassifier:
    """获取 QueryClassifier 单例"""
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier.get_instance()
    return _classifier


if __name__ == '__main__':
    # 离线训练/推理演示：需先准备训练数据并配置 .env.local
    import sys

    sys.path.insert(0, str(__file__).split("/backend/")[0])
    from dotenv import load_dotenv

    load_dotenv(".env.local")

    # ── 步骤一：微调模型（首次运行，后续跳过）──────────────────
    # 显式传入基座路径 → 加载基座 MiniLM（settings.classifier_model_path）
    # 注意：训练数据 backend/core/data/training_data.jsonl 为占位路径，
    # 请替换为实际标注数据后再运行。
    qc_base = QueryClassifier(
        model_path=os.path.join(backend_path, get_settings().classifier_model_path)
    )
    qc_base.train(
        data_path=os.path.join(backend_path, "backend/core/data/training_data.jsonl"),
        output_dir=os.path.join(backend_path, get_settings().finetuned_classifier_path),
        epochs=8,
        batch_size=64,
    )
    # 输出：✅ 微调完成，模型保存到：models/classifier/finetuned

    # ── 步骤二：加载微调模型，测试推理 ─────────────────────────
    qc = QueryClassifier()  # 无参 → 加载微调后模型

    test_cases = [
        # 通用问题：LLM 自身知识已覆盖，预期 general
        ("什么是数字机床？", "general"),
        ("Python 中 list 和 tuple 有什么区别？", "general"),
        # 专业问题：涉及设备/运维专属内容，预期 specialized
        ("设备报 E001 电机过载怎么排查？", "specialized"),
        ("诊断树三轨匹配的模糊匹配阈值是多少？", "specialized"),
    ]

    print(f"\n{'Query':<44} {'预期':<12} {'预测':<12} {'置信度'}")
    print("-" * 78)
    for text, expected in test_cases:
        label, conf = qc.classify(text)
        mark = "✓" if label == expected else "✗"
        print(f"{text:<44} {expected:<12} {label:<12} {conf:.4f}  {mark}")
