from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader


def save_into_faiss():
    with open("real_estate_sales_data.txt") as f:
        real_estate_sales = f.read()

    text_splitter = CharacterTextSplitter(
        separator=r'\d+\.',
        chunk_size=100,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=True,
    )
    docs = text_splitter.create_documents([real_estate_sales])


    # OpenAI Embedding 模型
    embeddings = OpenAIEmbeddings()
    # FAISS 向量数据库，使用 docs 的向量作为初始化存储
    db = FAISS.from_documents(docs, embeddings)
    # 构造提问 Query
    query = "小区吵不吵"
    # 在 Faiss 中进行相似度搜索，找出与 query 最相似结果
    docs = db.similarity_search(query)
    # 输出 Faiss 中最相似结果
    print(docs[0].page_content)

    db.save_local("real_estates_sale")

def read_from_faiss():
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings)
    query = "What did the president say about Ketanji Brown Jackson"
    docs = new_db.similarity_search(query)
    print(docs[0].page_content)

if __name__ == "__main__":
    save_into_faiss()