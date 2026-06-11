"""DeepSeek LLM integration."""

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from config import LLM_TEMPERATURE, LLM_MAX_TOKENS


_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

PROMPT_TEMPLATE = (
    "你是一位专业的保险理赔分析师。请根据以下保单上下文，回答用户的问题。\n\n"
    "## 上下文（保单片段）\n{context}\n\n"
    "## 保单元数据\n{metadata}\n\n"
    "## 用户问题\n{question}\n\n"
    "## 回答要求\n"
    "1. 如果上下文明确包含答案，请引用原文并注明页码。\n"
    "2. 涉及变更时，必须列出变更前后对比表格。\n"
    "3. 如果上下文信息不足，请如实说明，不要编造。\n"
    "4. 使用中文、列表格式回答，清晰易读。\n\n"
    "## 答案"
)


def answer(context: str, metadata: str, question: str) -> str:
    """Call DeepSeek API with context + metadata."""
    prompt = PROMPT_TEMPLATE.format(context=context, metadata=metadata, question=question)
    try:
        resp = _client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[LLM 调用失败] {e}"


def check_connection() -> bool:
    """Verify DeepSeek API connectivity."""
    try:
        models = _client.models.list()
        return True
    except Exception:
        return False
