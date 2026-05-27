from hello_agents import HelloAgentsLLM, ToolRegistry, ReActAgent
from hello_agents import ReflectionAgent, PlanAndSolveAgent
from hello_agents.tools import CalculatorTool
from my_agent_react import MyReActAgent

llm = HelloAgentsLLM(
    provider="ollama",
    model="qwen2.5:3b",  # 需与 `ollama run` 指定的模型一致
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # 本地服务同样不需要真实 Key
)
tools = ToolRegistry()
tools.register_tool(CalculatorTool())

# ======= 1 ReAct ==========
# agent = ReActAgent(name="AI", llm=llm, system_prompt="你是助手", tool_registry=tools)
# agent = MyReActAgent(name="AI", llm=llm, system_prompt="你是助手", tool_registry=tools)
# response = agent.run("3+5*18")
# print(response)
# print(f"历史消息数: {len(agent.get_history())}")


# ======= 2 Reflection ==========
# general_agent = ReflectionAgent(name="我的反思助手", llm=llm)
# code_agent = ReflectionAgent(
#     name="我的代码生成助手",
#     llm=llm,
#     custom_prompts={
#         "initial": "你是Python专家，请编写函数:{task}",
#         "reflect": "请审查代码的算法效率:\n任务:{task}\n代码:{content}",
#         "refine": "请根据反馈优化代码:\n任务:{task}\n反馈:{feedback}",
#     },
# )
# result = general_agent.run("写一篇关于人工智能发展历程的简短文章")
# result = code_agent.run("写一个高效的排序")
# print(f"最终结果: {result}")


# ======= 1 ReAct ==========
# 创建自定义PlanAndSolveAgent
agent = PlanAndSolveAgent(name="我的规划执行助手", llm=llm)
question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
result = agent.run(question)
print(f"\n最终结果: {result}")


print(f"对话历史: {len(agent.get_history())} 条消息")
