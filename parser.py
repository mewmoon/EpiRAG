from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnableLambda

json_parser = JsonOutputParser()
str_parser = StrOutputParser()

# first_prompt = PromptTemplate.from_template(
#     "我的邻居姓{lastname}, 刚生了{gender}, 你帮我起个名字，严格以键值对形式给出，键为name，值为名字，不要包含其他内容。"
# )
first_prompt = PromptTemplate.from_template(
    "我的邻居姓{lastname}, 刚生了{gender}, 你帮我起个名字，仅仅给出名字就行，不要包含其他内容。"
)

second_prompt = PromptTemplate.from_template(
    "{name} 为我的名字，我想把第二个字改了,给出两个备选"
)


model = ChatTongyi(model="qwen-max")
# chain = first_prompt | model | json_parser | second_prompt | model | str_parser
my_func = RunnableLambda(lambda ai_msg: {"name": ai_msg.content})


def print_ans(ai_msg):
    print("模型输出：", ai_msg.content)
    return ai_msg


chain = first_prompt | model | print_ans | my_func | second_prompt | model | str_parser
res = chain.invoke(input={"lastname": "李", "gender": "女儿"})
print(res)
