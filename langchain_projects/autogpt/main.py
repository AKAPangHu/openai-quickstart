import os
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain_community.embeddings import OpenAIEmbeddings
import gradio as gr

from langchain_experimental.autonomous_agents import AutoGPT
from langchain.chat_models import ChatOpenAI

import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

os.environ["SERPAPI_API_KEY"] = "1409f253431dc3b70eef51617ea1aa8a439709192ebd51343457225b51f6835a"


def test_serpapi(text):
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted "
                        "questions",
        ),
        WriteFileTool(),
        ReadFileTool()
    ]

    embeddings_model = OpenAIEmbeddings()

    # OpenAI Embedding 向量维数
    embedding_size = 1536
    # 使用 Faiss 的 IndexFlatL2 索引
    index = faiss.IndexFlatL2(embedding_size)
    # 实例化 Faiss 向量数据库
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    agent = AutoGPT.from_llm_and_tools(
        ai_name="Jarvis",
        ai_role="Assistant",
        tools=tools,
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, verbose=True),
        memory=vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.8}),  # 实例化 Faiss 的 VectorStoreRetriever
    )

    agent.chain.verbose = True

    return agent.run([text])


def greet(text):
    return test_serpapi(text)


def init_gradio():
    demo = gr.Interface(
        fn=greet,
        title="Auto-GPT",
        description="一个自动拆解任务，回答的智能机器人！",
        inputs=["text"],
        outputs=["text"],
    )

    demo.launch()


if __name__ == "__main__":
    # test_serpapi()
    init_gradio()