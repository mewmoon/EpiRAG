from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

csv_loader = CSVLoader(
    "documents/stu.csv",
    encoding="utf-8",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        # "fieldnames": ["a", "b", "c"],
    },
)

json_loader = JSONLoader(
    file_path="documents/stu.jsonl",
    jq_schema=".name",
    text_content=True,  # 是否抽取字符串
    json_lines=True,
)

text_loader = TextLoader(file_path="documents/story.txt", encoding="utf-8")
docs = text_loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    separators=["\n\n", "。", "！", "？", "?", "!", "."],
    length_function=len,
)
split_docs = splitter.split_documents(docs)

pdf_loader = PyPDFLoader(
    file_path="documents/book.pdf",
    mode="page",  # single变成一个
)

i = 0
for d in pdf_loader.lazy_load():
    print(d)
    i += 1
    print("=" * 20, i, "=" * 20)
