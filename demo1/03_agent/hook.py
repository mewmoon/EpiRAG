from typing import Annotated, TypedDict
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


# 1. 定义状态 (Middleware 传递的数据)
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def get_weather(city: str):
    """查询天气的工具"""
    return "晴天 30°"


tools = [get_weather]
model = ChatTongyi(model="qwen3-max").bind_tools(tools)


# 3. 定义中间件逻辑 (Node)
def call_model(state: AgentState):
    print("--- [Middleware] 正在进入模型推理 ---")
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# 4. 构建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

# 添加边 (流程控制)
workflow.set_entry_point("agent")


# 条件边：如果模型决定调用工具，则去 tools，否则结束
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        print("--- [Middleware] 拦截：检测到工具调用 ---")
        return "tools"
    return END


workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")  # 工具执行完返回 agent

# 编译
app = workflow.compile()

# 执行
inputs = {"messages": [HumanMessage(content="明天南京的天气怎么样？")]}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"节点 {key} 执行完毕")
