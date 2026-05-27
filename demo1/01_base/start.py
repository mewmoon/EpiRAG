from langchain_community.llms.tongyi import Tongyi
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

model = Tongyi(model="qwen-max", temperature=0.9)
chat_model = ChatTongyi(model="qwen3-max", temperature=0.9)
embed = DashScopeEmbeddings()

messages = [
    SystemMessage(content="你是一个诗人"),
    HumanMessage(content="写一首诗"),
    AIMessage(content="锄禾日当午，汗滴禾下土"),
    HumanMessage(content="不错，继续"),
]

# res = model.invoke("请介绍一下自己")
# res = chat_model.stream(input=messages)
# for chunk in res:
#     print(chunk, end=" ", flush=True)

print(embed.embed_query("我喜欢你"))
print(embed.embed_documents(["我喜欢你", "晚上吃啥"]))
