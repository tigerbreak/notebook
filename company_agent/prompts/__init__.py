"""提示词模块"""
import os
from typing import Optional


def load_prompt(filename: str) -> str:
    """加载提示词文件"""
    prompt_dir = os.path.dirname(__file__)
    with open(os.path.join(prompt_dir, filename), "r", encoding="utf-8") as f:
        return f.read()


SYSTEM_PROMPT = load_prompt("system.md")
NEXT_STEP_PROMPT = load_prompt("next_step.md")


def build_next_step_prompt(user_query: str, search_history: Optional[str] = None) -> str:
    """
    构建下一步的决策提示词。
    对应 RAGFlow: rag/prompts/next_step.md
    """
    if search_history:
        history_text = f"## 已获取的信息\n\n{search_history}"
    else:
        history_text = "你还没有获取任何信息。"
    
    return NEXT_STEP_PROMPT.format(
        user_query=user_query,
        search_history=history_text
    )