import re

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import hashlib

from sympy import content
import config_rag as config
from history_service import get_history


class RagService:
    def __init__(self):
        self.embeddings = DashScopeEmbeddings()
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=config.persist_directory,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        self.model = Tongyi(model="qwen-max")

        # 使用 ChatPromptTemplate 来处理消息占位符
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "回答用户提问: 若资料中未提及相关信息，请直接回答“资料中未找到相关答案”，不要胡编乱造。",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("system", "【背景资料】:\n{context}\n\n【用户提问】: {input}"),
            ]
        )

        self.base_chain = (
            {
                "context": RunnableLambda(self.format_for_rag)
                | self.retriever
                | self._format_docs,
                "input": RunnablePassthrough(),
            }
            | RunnableLambda(self.format_for_prompt)
            | RunnableLambda(self.print_in)
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        self.history_chain = RunnableWithMessageHistory(
            self.base_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _format_docs(self, docs):
        return "\n\n".join([d.page_content for d in docs])

    def _generate_id(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    @staticmethod
    def format_for_rag(value):
        return value["input"]

    @staticmethod
    def format_for_prompt(value):
        return {
            "input": value["input"]["input"],
            "history": value["input"]["history"],
            "context": value["context"],
        }

    @staticmethod
    def print_in(text):
        print("====================")
        print(text, type(text))
        return text

    def upload_text(self, text: str, metadata: dict = None):
        doc = Document(page_content=text, metadata=metadata or {})
        docs = self.splitter.split_documents([doc])
        ids = [self._generate_id(d.page_content) for d in docs]
        self.vector_store.add_documents(docs, ids=ids)
        return True

    def answer_question(self, question):
        return self.history_chain.invoke(
            {"input": question}, config=config.session_config
        )

    def stream_answer(self, question):
        return self.history_chain.stream(
            {"input": question}, config=config.session_config
        )


if __name__ == "__main__":
    # --- 正确的调用方式 ---
    rag_service = RagService()
    res = rag_service.history_chain.invoke(
        {"input": "小明考了多少分"}, config=config.session_config
    )
    print(res)
