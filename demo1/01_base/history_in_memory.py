from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
import os
from history_in_file import FileChatHistory

chats = {}


# def get_history(session_id):
#     if session_id not in chats:
#         chats[session_id] = InMemoryChatMessageHistory()
#     return chats[session_id]


def get_history(session_id):
    return FileChatHistory(session_id, "chat_history")


def print_prompt(full_prompt):
    print("=" * 20)
    print(full_prompt.to_string())
    return full_prompt


model = ChatTongyi(model="qwen3-max")
# prompt = PromptTemplate.from_template(
#     "根据历史会话信息回答问题。历史会话{chat_history}，用户问题{input}"
# )
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "根据历史会话信息回答问题。对话历史："),
        MessagesPlaceholder("chat_history"),
        ("human", "请回答：{question}"),
    ]
)
str_parser = StrOutputParser()
base_chain = prompt | print_prompt | model | str_parser

conversation_chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

if __name__ == "__main__":
    session_config = {"configurable": {"session_id": "user001"}}
    # res = conversation_chain.invoke({"question": "小明有两只猫"}, session_config)
    # print("\n第一次执行", res)
    # res = conversation_chain.invoke({"question": "小张有1只狗"}, session_config)
    # print("\n第二次执行", res)
    res = conversation_chain.invoke({"question": "总共有几只宠物"}, session_config)
    print("\n第三次执行", res)

    print("=" * 10, "History", "=" * 10)
    print(get_history(session_config["configurable"]["session_id"]))
