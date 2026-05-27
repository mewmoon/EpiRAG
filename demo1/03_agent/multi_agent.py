import openai
import os

# 假设你已经设置好了 API Key
client = openai.OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 使用你的环境变量
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 核心适配地址
)


def get_response(messages):
    response = client.chat.completions.create(model="qwen-max", messages=messages)
    return response.choices[0].message.content


# 1. 设定角色
coder_agent_prompt = "你是一名专业的量化交易策略编写员。请提供一个简单的均线交叉策略。"
reviewer_agent_prompt = "你是一名严格的风险控制官。请审查对方提供的交易策略，指出风险点，并给出具体的改进建议。"

# 2. 初始对话记录
conversation = [
    {"role": "system", "content": coder_agent_prompt},
    {"role": "user", "content": "请写一个简单的均线交叉策略代码框架。"},
]

# 3. Agent 1 (Coder) 输出
coder_output = get_response(conversation)
print(f"--- Coder Agent ---\n{coder_output}\n")

# 4. 传递给 Agent 2 (Reviewer)
conversation.append({"role": "assistant", "content": coder_output})
conversation.append({"role": "system", "content": reviewer_agent_prompt})

# 5. Agent 2 (Reviewer) 输出
reviewer_output = get_response(conversation)
print(f"--- Reviewer Agent ---\n{reviewer_output}\n")
