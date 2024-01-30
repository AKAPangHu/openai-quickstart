import tiktoken
from langchain_community.document_loaders import BiliBiliLoader
from bilibili_api import Credential, sync


def count_tokens(string: str, model: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


if __name__ == "__main__":
    loader = BiliBiliLoader(["https://www.bilibili.com/video/BV1xt411o7Xu/"])
    doc = loader.load()

    credential = Credential(sessdata="你的 SESSDATA", bili_jct="你的 bili_jct", buvid3="你的 buvid3",
                            dedeuserid="你的 DedeUserID", ac_time_value="你的 ac_time_value")

    print(sync(credential.check_refresh()))
    sync(credential.refresh())

    count = count_tokens(doc[0].page_content, "gpt-3.5-turbo")
    print(f'count: {count}, len: {len(doc[0].page_content)}, page_content: {doc[0].page_content}')
