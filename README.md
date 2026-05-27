pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install langchain langchain-community langchain-ollama dashscope chromadb
pip install jq pypdf streamlit

### run Rag
cd 02_rag
streamlit run app.py

pip install autogen-agentchat autogen-ext