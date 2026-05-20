from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个诗人"),
        MessagesPlaceholder("history"),
        ("human", "这是同一个作者吗？"),
    ]
)

history_data = [
    ("human", "你来作一首唐诗"),
    ("assistant", "床前明月光，疑是地上霜，举头望明月，低头思故乡。"),
    ("human", "好诗，再来一首"),
    ("assistant", "锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
]


prompt_text = prompt.invoke({"history": history_data}).to_string()
print(prompt_text)


model = ChatTongyi(model="qwen3-max")

print(model.invoke(prompt_text))
