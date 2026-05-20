from langchain_community.llms.tongyi import Tongyi

model = Tongyi(model="qwen-max", temperature=0.9)

res = model.invoke("请介绍一下自己")

print(res)
