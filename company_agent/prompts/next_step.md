# 下一步决策提示词

对应 RAGFlow: rag/prompts/next_step.md

用户问题: {user_query}

{search_history}

## 请思考

1. 用户想要了解什么？
2. 我已经有什么信息？还需要什么？
3. 应该调用哪个 API 来获取缺少的信息？

然后：
- 调用合适的工具来获取更多信息
- 或者，如果信息已经足够，给出最终回答