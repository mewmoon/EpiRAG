from langchain_core.prompts import PromptTemplate
from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms.tongyi import Tongyi
from langchain_core.output_parsers import StrOutputParser
from typing import List

model = Tongyi(model="qwen-max")
prompt = PromptTemplate.from_template(
    "根据相关背景简要回答问题，不要胡编乱造。" "相关知识：{context}" "用户提问：{input}"
)

embedings = DashScopeEmbeddings()
vector_store = Chroma(
    collection_name="my_collection_1",
    embedding_function=embedings,
    persist_directory="chroma_db",
)
retriver = vector_store.as_retriever(search_kwargs={"k": 2})


# vector_store.add_texts(
#     texts=["减肥有利于身体健康", "每天一个苹果，医生远离我", "香蕉是黄色的"],
#     metadatas=[{"source": "黑马"}, {"source": "抓马"}, {"source": "黑马"}],
# )
# res = vector_store.similarity_search("水果", k=2)
def print_prompt(prompt):
    print("=" * 20)
    print(prompt)
    print("=" * 20)
    return prompt


def format_docs(docs):
    if not docs:
        return "无参考资料"
    re = "["
    for doc in docs:
        re += doc.page_content
        re += ";"
    re += "]"
    return re


chain = (
    {"input": RunnablePassthrough(), "context": retriver | format_docs}
    | prompt
    | print_prompt
    | model
    | StrOutputParser()
)
res = chain.invoke("怎么健康?")
print(res)
