import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tavily import TavilyClient

# 自动加载环境变量（如 .env 文件）
load_dotenv()


# ==========================================
# 1. 定义状态结构 (State)
# ==========================================
class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str  # 经过LLM理解后的用户需求总结
    search_query: str  # 优化后用于Tavily API的搜索查询
    search_results: str  # Tavily搜索返回的结果
    final_answer: str  # 最终生成的答案
    step: str  # 标记当前步骤


# ==========================================
# 2. 初始化客户端与模型
# ==========================================
# 确保在运行前设置了环境变量 DASHSCOPE_API_KEY 和 TAVILY_API_KEY
llm = ChatTongyi(model="qwen3-max")
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ==========================================
# 3. 定义节点 (Nodes)
# ==========================================


def understand_query_node(state: SearchState) -> dict:
    """步骤1：理解用户查询并生成搜索关键词"""
    # 获取用户发送的最后一条消息
    user_message = state["messages"][-1].content

    understand_prompt = f"""分析用户的查询："{user_message}"
        请完成三个任务：
        1. 简洁总结用户想要了解什么
        2. 判断是否需要联网查询，如果不许要联网，则输出no_need_search，否则生成最适合搜索引擎的关键词（中英文均可，要精准，不要带任何标点符号）。

        格式必须严格如下：
        理解：[用户需求总结]
        搜索：[no_need_search/最佳搜索关键词]"""

    response = llm.invoke([SystemMessage(content=understand_prompt)])
    response_text = response.content

    # 解析LLM的输出，提取搜索关键词
    search_query = user_message  # 默认使用原始查询作为保底
    if "搜索：" in response_text:
        search_query = response_text.split("搜索：")[1].strip()
    elif "搜索:" in response_text:
        search_query = response_text.split("搜索:")[1].strip()

    print("==========\n", response_text, "\n=================", sep="")
    return {
        "user_query": response_text,
        "search_query": search_query,
        "step": "understood",
        "messages": [AIMessage(content=f"🔍 思索结果：{search_query}")],
    }


def tavily_search_node(state: SearchState) -> dict:
    """步骤2：使用Tavily API进行真实搜索"""
    search_query = state["search_query"]
    try:
        print(f"\n[Node: Search] 正在调用 Tavily 搜索: '{search_query}'...")
        response = tavily_client.search(
            query=search_query, search_depth="basic", max_results=3, include_answer=True
        )

        # 补全逻辑：解析并格式化搜索结果
        results_list = []
        if response.get("answer"):
            results_list.append(f"Tavily 智能摘要: {response['answer']}\n")

        for i, res in enumerate(response.get("results", []), 1):
            results_list.append(
                f"[{i}] 标题: {res.get('title')}\n链接: {res.get('url')}\n摘要: {res.get('content')}\n---"
            )

        search_results = "\n".join(results_list)
        if not search_results.strip():
            search_results = "未找到相关搜索结果。"

        return {
            "search_results": search_results,
            "step": "searched",
            "messages": [
                AIMessage(content="✅ 搜索完成！正在为您整理并生成深度答案...")
            ],
        }
    except Exception as e:
        print(f"[Node: Search] 搜索发生错误: {e}")
        return {
            "search_results": f"搜索失败：{e}",
            "step": "search_failed",
            "messages": [
                AIMessage(
                    content="❌ 搜索遇到了一些技术问题，我将转为离线知识为您解答。"
                )
            ],
        }


def generate_answer_node(state: SearchState) -> dict:
    """步骤3：基于搜索结果生成最终答案"""
    # 提取用户的原始提问（从消息列表的最后一条 HumanMessage 或初始消息中获取）
    original_question = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            original_question = msg.content
            break

    if state["step"] == "search_failed":
        # 如果搜索失败，执行回退策略，基于LLM自身知识回答
        fallback_prompt = f"由于网络或API限制，实时搜索暂时不可用。请完全基于您自身的知识，尽可能准确地回答用户的问题。\n\n用户问题：{original_question}"
        response = llm.invoke([SystemMessage(content=fallback_prompt)])
    else:
        # 搜索成功，基于搜索结果生成答案
        answer_prompt = f"""请结合给出的实时搜索结果，为用户提供一个结构清晰、详实、准确的最终答案。如果在搜索结果中发现冲突，请为用户客观客观指出。

            用户问题：{original_question}

            搜索结果：
            {state['search_results']}

            请直接输出你给用户的回答，保持语气礼貌专业："""
        response = llm.invoke([SystemMessage(content=answer_prompt)])

    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [AIMessage(content=response.content)],
    }


# ==========================================
# 4. 构建与编译图 (Graph)
# ==========================================
def create_search_assistant():
    workflow = StateGraph(SearchState)

    # 添加节点
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)

    # 设置线性流程
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)

    # 编译图，并加入内存检查点（实现多轮对话记忆基础）
    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


def routing_logic(state: SearchState) -> str:
    """根据 understand 节点的分析结果，自主选择下一条路径"""
    user_analysis = state["user_query"]

    if "no_need_search" in user_analysis or state["search_query"] == "无":
        print("🤖 Agent 决策：这是一个常识/闲聊问题，无需联网，直接回答。")
        return "direct_answer"
    else:
        print("🤖 Agent 决策：需要最新实时信息，触发联网搜索。")
        return "go_search"


# 2. 修改：编译图时的连线逻辑
def create_true_agent():
    workflow = StateGraph(SearchState)

    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)

    # 连线升级：
    workflow.add_edge(START, "understand")

    # 从 understand 出来时，不写死下一步，而是交给 routing_logic 动态决定
    workflow.add_conditional_edges(
        "understand",
        routing_logic,
        {
            "go_search": "search",  # 如果返回 go_search，走 search 节点
            "direct_answer": "answer",  # 如果返回 direct_answer，直接跳到 answer 节点
        },
    )

    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)

    memory = InMemorySaver()
    return workflow.compile(checkpointer=memory)


# ==========================================

# 5. 测试运行入口
# ==========================================
if __name__ == "__main__":
    # 创建图应用
    # app = create_search_assistant()
    app = create_true_agent()

    # 配置线程 ID（LangGraph 跟踪会话所必须）
    config = {"configurable": {"thread_id": "test_session_123"}}

    # 用户输入测试问题
    user_input = "2026年，中国重大新闻有哪些？"
    # user_input = "1+1=?"
    print(f"👤 用户提问: {user_input}")

    # 构造初始状态输入
    # initial_input = {"messages": [HumanMessage(content=user_input)]}
    # 构造初始状态输入时，把所有字段都初始化
    initial_input = {
        "messages": [HumanMessage(content=user_input)],
        "user_query": "",
        "search_query": "",
        "search_results": "",
        "final_answer": "",
        "step": "init",
    }

    # 流式输出图的执行节点和状态更新
    print("\n🚀 开始执行智能体工作流...")
    for event in app.stream(initial_input, config=config, stream_mode="values"):
        # 打印当前产生的最后一条消息，以此观察中间思考和执行过程
        if "messages" in event and event["messages"]:
            last_msg = event["messages"][-1]
            # 避免重复打印用户自己的输入
            if not isinstance(last_msg, HumanMessage):
                print(f"🤖 Agent: {last_msg.content}")

    print("\n🏁 工作流执行完毕。")
