from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.tools import tool


@tool(description="查询天气的工具")
def get_weather(city):
    return "晴天 30°"


agent = create_agent(
    model=ChatTongyi(model="qwen3-max"),
    system_prompt="你是一个聊天助手",
    tools=[get_weather],
)

for chunk in agent.stream(
    {"messages": [("user", "明天南京的天气怎么样?我穿啥合适？")]}, stream_mode="values"
):
    last_message = chunk["messages"][-1]
    if last_message.content:
        print(type(last_message).__name__, last_message.content)
    try:
        if last_message.tool_calls:
            print(
                f"{type(last_message).__name__ } 工具调用 {[tc['name'] for tc in last_message.tool_calls]}"
            )
    except AttributeError:
        pass
