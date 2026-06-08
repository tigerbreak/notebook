
"""prompts 包 — 从 .md 文件加载提示词"""
import os

_prompt_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_prompt_dir, "system.md"), "r", encoding="utf-8") as f_p:
    SYSTEM_PROMPT = f_p.read()

with open(os.path.join(_prompt_dir, "next_step.md"), "r", encoding="utf-8") as f_p:
    NEXT_STEP_PROMPT = f_p.read()


def build_next_step_prompt(user_query: str, search_history: str = "") -> str:
    """构建每一步的提示词"""
    prompt = NEXT_STEP_PROMPT.format(
        user_query=user_query,
        search_history=search_history.strip(),
    )
    return prompt


__all__ = ["SYSTEM_PROMPT", "NEXT_STEP_PROMPT", "build_next_step_prompt"]
