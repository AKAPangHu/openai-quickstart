import time

from enum import unique, Enum
import gradio as gr

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from langchain_projects.sales_chatbot.faiss_services.faiss_service import save_into_faiss

ENABLE_CHAT = False
BOT = None


@unique
class SceneEnum(Enum):
    房产 = "real_estate"
    iPhone = "iphone"
    英语培训 = "english_training"


def initialize_sales_bot(vector_store_dir: str):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings())
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    global BOT
    BOT = RetrievalQA.from_chain_type(llm,
                                      retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    BOT.return_source_documents = True

    return BOT


def chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"


def add_text(history, text):
    # print(f"add_text-[history]{history}")
    # print(f"add_text-[text]{text}")

    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def change_scene(choice):
    # print(f"change_scene-[choice]{choice}")
    vector_store_dir = choice + "_sales"
    initialize_sales_bot(vector_store_dir)
    return []


def change_enable_chat(enable):
    global ENABLE_CHAT
    ENABLE_CHAT = enable


def bot(history, text):
    # print(f"bot-[history]{history}")
    # print(f"bot-[text]{text}")
    query = history[-1][0]

    ans = BOT({"query": query})

    response = "这个问题我要问问领导"
    if ans["source_documents"] or ENABLE_CHAT:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        response = ans["result"]

    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history


def launch_gradio_by_blocks():
    with gr.Blocks(title="销售机器人") as blocks:
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    scene_radio = gr.Radio(
                        [(member.name, member.value) for member in SceneEnum],
                        label="切换场景",
                        info="选择一个要咨询的场景",
                        value=SceneEnum.房产.name,
                    )
                    enable_chat_checkbox = gr.Checkbox(
                        label="激活 AI", info="通过 AI 更好的回答问题", value=ENABLE_CHAT
                    )
            with gr.Column(scale=4):
                chatbot = gr.Chatbot([], elem_id="chatbot", bubble_full_width=False)
                with gr.Row():
                    txt = gr.Textbox(
                        scale=4,
                        show_label=False,
                        placeholder=" 请输入你想咨询的问题",
                        container=False,
                    )
                    # btn = gr.Button("Submit")
        txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=True).then(
            bot, chatbot, chatbot
        )
        txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

        scene_radio.change(fn=change_scene, inputs=scene_radio, outputs=chatbot)

        enable_chat_checkbox.change(fn=change_enable_chat, inputs=enable_chat_checkbox)

        # btn.click(add_text, [chatbot, txt], [chatbot, txt], queue=True).then(bot, chatbot,chatbot)

    blocks.queue(max_size=10)
    blocks.launch(share=True, server_name="0.0.0.0", server_port=7861)


def launch_gradio():
    demo = gr.ChatInterface(
        fn=chat,
        title="房产销售",
        chatbot=gr.Chatbot(height=600),
    )
    demo.launch(share=True, server_name="0.0.0.0")


def initialize_faiss_data():
    for e in SceneEnum:
        save_into_faiss(e.value)


if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot("real_estate_sales")
    # 启动 Gradio 服务
    launch_gradio_by_blocks()
