import streamlit as st
from rag_service import RagService
from history_service import get_history
from langchain_core.messages import AIMessage, HumanMessage
import config_rag as config

if "rag" not in st.session_state:
    st.session_state.rag = RagService()

st.set_page_config(page_title="阅读助理", layout="wide")
st.title("📚 智能阅读助理 (RAG)")

# 2. 侧边栏：文档上传与处理
with st.sidebar:
    st.header("知识库管理")
    uploaded_file = st.file_uploader("上传文档 (TXT)", type=["txt"])
    if uploaded_file and st.button("存入知识库"):
        text = uploaded_file.read().decode("utf-8")
        if st.session_state.rag.upload_text(
            text, metadata={"source": uploaded_file.name}
        ):
            st.success(f"已上传: {uploaded_file.name}")
        else:
            st.error(f"上传失败")

# 3. 主聊天界面
if "messages" not in st.session_state:
    history_obj = get_history(config.session_config["configurable"]["session_id"])
    st.session_state.messages = []
    for msg in history_obj.messages:
        if isinstance(msg, HumanMessage):
            st.session_state.messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            st.session_state.messages.append(
                {"role": "assistant", "content": msg.content}
            )

# 显示历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("有什么想问的吗？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # 创建占位符
        full_response = ""
        response_generator = st.session_state.rag.stream_answer(prompt)
        full_response = st.write_stream(response_generator)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
