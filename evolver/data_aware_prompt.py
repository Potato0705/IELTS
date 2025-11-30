# evolver/data_aware_prompt.py
from __future__ import annotations
import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# ========= 加载数据集特定的模板 =========

def load_dataset_templates(dataset_name: str = None) -> Dict[str, Any]:
    """根据数据集名称加载对应的 prompt 模板"""
    if dataset_name is None:
        dataset_name = os.getenv("DATASET_NAME", "ielts_chillies")
    
    templates_path = Path(__file__).parent / "prompt_templates.json"
    
    with open(templates_path, 'r', encoding='utf-8') as f:
        all_templates = json.load(f)
    
    if dataset_name not in all_templates:
        print(f"⚠️  未找到数据集 '{dataset_name}' 的模板，使用默认 'ielts_chillies'")
        dataset_name = "ielts_chillies"
    
    return all_templates[dataset_name]

# 全局加载当前数据集的模板
_CURRENT_TEMPLATES = load_dataset_templates()

# ========= Genome =========

@dataclass(frozen=True)
class PromptGenome:
    instruction_id: int = 0
    instruction_text: Optional[str] = None  # ✅ LLM 生成的新模板文本（优先级最高）
    strictness: int = 1
    output_format: str = "scalar"

    # ===== ICL 策略：两种模式 =====
    use_icl_indices: bool = False  # 🔥 开关：True=使用索引列表，False=使用策略
    
    # 模式1：策略驱动（旧方式）
    icl_strategy: str = "random"
    k_shots: int = 0
    
    # 模式2：索引驱动（新方式）
    icl_indices: Optional[tuple] = None  # 🔥 训练集索引列表，如 (12, 45, 78, ...)
    
    # ===== 其他 =====
    rag_strategy: str = "none"
    use_summary: bool = False

    use_teacher: bool = False
    teacher_weight: float = 0.0


# ========= Instruction templates =========
# 动态加载：根据当前数据集获取模板
def get_instruction_templates() -> Dict[int, str]:
    """获取当前数据集的 instruction 模板"""
    templates = _CURRENT_TEMPLATES.get("instruction_templates", {})
    return {int(k): v for k, v in templates.items()}

def get_score_range() -> Dict[str, float]:
    """获取当前数据集的评分范围"""
    return _CURRENT_TEMPLATES.get("score_range", {"min": 0, "max": 9, "step": 0.5})

def calibrate_score(raw_score: float) -> float:
    """根据当前数据集的评分范围校准分数"""
    score_range = get_score_range()
    min_score = score_range.get("min", 0)
    max_score = score_range.get("max", 9)
    step = score_range.get("step", 0.5)
    
    # 限制在范围内
    score = max(min_score, min(max_score, raw_score))
    
    # 四舍五入到最近的步长
    if step == 1:
        score = round(score)
    elif step == 0.5:
        score = round(score * 2) / 2.0
    else:
        score = round(score / step) * step
    
    return float(score)

INSTRUCTION_TEMPLATES: Dict[int, str] = get_instruction_templates()


STRICTNESS_CLAUSES: Dict[int, str] = {
    0: "Be fair and neutral in scoring.",
    1: "Be strict but fair, avoid score inflation.",
}

def get_output_scalar_clause() -> str:
    """根据当前数据集的评分范围生成输出指令"""
    score_range = get_score_range()
    min_score = score_range.get("min", 0)
    max_score = score_range.get("max", 9)
    step = score_range.get("step", 0.5)
    
    if step == 1:
        example = f"{int((min_score + max_score) / 2)}"
        step_desc = "whole numbers"
    elif step == 0.5:
        example = f"{(min_score + max_score) / 2:.1f}"
        step_desc = "0.5 steps"
    else:
        example = f"{(min_score + max_score) / 2:.1f}"
        step_desc = f"{step} steps"
    
    return (
        f"Output ONLY the final overall score as a single number "
        f"from {min_score} to {max_score} in {step_desc} (e.g., {example}). "
        f"Do NOT output any explanation, text, or symbols."
    )

UNDERLEN_PENALTY = "If the essay is clearly under 250 words, lower the score by at least 0.5."


def _format_example(ex: Dict[str, Any], max_len: int = 1200) -> str:
    """Few-shot example formatter."""
    p = str(ex.get("prompt", "")).strip()
    e = str(ex.get("essay", "")).strip()
    b = float(ex.get("band", 5.0))

    if len(e) > max_len:
        e = e[:max_len] + " ..."

    return (
        "=== Example ===\n"
        f"Prompt:\n{p}\n\n"
        f"Essay:\n{e}\n\n"
        f"Score: {b:.1f}\n"
    )


def build_full_prompt(
    genome: PromptGenome,
    essay: str,
    icl_examples: Optional[List[Dict[str, Any]]] = None,
    rag_examples: Optional[List[Dict[str, Any]]] = None,
    summary_text: Optional[str] = None,
) -> str:
    """
    Build structured prompt:
    [instruction + strictness + output-format]
    + few-shot examples (ICL)
    + RAG examples (stub)
    + summary (stub)
    + target essay
    """
    # 动态获取当前数据集的模板
    templates = get_instruction_templates()
    
    # ✅ instruction_text 优先
    if genome.instruction_text and str(genome.instruction_text).strip():
        instruction = str(genome.instruction_text).strip()
    else:
        instruction = templates.get(genome.instruction_id, templates.get(0, ""))

    strictness = STRICTNESS_CLAUSES.get(genome.strictness, STRICTNESS_CLAUSES[1])
    output_clause = get_output_scalar_clause()

    parts: List[str] = []
    parts.append(instruction + " " + strictness + " " + UNDERLEN_PENALTY)
    parts.append(output_clause)

    if icl_examples:
        for ex in icl_examples:
            parts.append(_format_example(ex))

    if rag_examples:
        for ex in rag_examples:
            parts.append(_format_example(ex))

    if summary_text:
        parts.append("=== Summary of the essay ===\n" + summary_text.strip())

    parts.append("=== Essay to score ===\n" + (essay or "").strip())
    parts.append("\nFinal overall band score:")

    return "\n\n".join(parts).strip()
