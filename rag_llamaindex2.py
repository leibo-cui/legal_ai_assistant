import json
import time
from pathlib import Path
from typing import List, Dict
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
import chromadb
from typing import AsyncGenerator



QA_TEMPLATE = (
    "<|im_start|>system\n"
    "你是一个专业的法律助手，请严格根据以下法律条文回答问题：\n"
    "相关法律条文：\n{context_str}\n<|im_end|>\n"
    "<|im_start|>user\n{query_str}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

response_template = PromptTemplate(QA_TEMPLATE)

############配置区###############
# class Config:
#     # EMBED_MODEL_PATH=r"E:\Export\.cache\huggingface\hub\models--BAAI--bge-small-zh-v1.5\snapshots\7999e1d3359715c523056ef9478215996d62a620"
#     # LLM_MODEL_PATH=r"E:\Export\.cache\huggingface\hub\models--Qwen--Qwen2.5-3B-Instruct\snapshots\aa8e72537993ba99e69dfaafa59ed015b17504d1"
#
#     EMBED_MODEL_PATH="BAAI/bge-small-zh-v1.5"
#     LLM_MODEL_PATH="qwen2.5:latest"
#
#     DATA_DIR=r"./data" # Document data
#     VECTOR_DB_PATH="./chroma_db"
#     PERSIST_DIR="./storage" # Store metadata
#
#     COLLECTION_NAME="chinese_labor_laws" #
#     TOP_K = 3 # Decide the number of item when execute


import yaml
from pathlib import Path


class Config:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 转换为字符串路径
        self.DATA_DIR = str(Path(config["paths"]["data_dir"]))
        self.VECTOR_DB_PATH = str(Path(config["paths"]["vector_db_path"]))
        self.PERSIST_DIR = str(Path(config["paths"]["persist_dir"]))

        # 映射配置到属性
        self.EMBED_MODEL_PATH = config["models"]["embed"]
        self.LLM_MODEL_PATH = config["models"]["llm"]

        # self.DATA_DIR = r"./data"
        # self.VECTOR_DB_PATH = "./chroma_db"
        # self.PERSIST_DIR = "./storage"
        self.COLLECTION_NAME = "chinese_labor_laws"  # 可保留或移至配置文件
        self.TOP_K = 3  # 可保留或移至配置文件


# 全局配置对象
CONFIG = Config()


############Initializate model###############
def init_models():
    """ Initialize model and verify """
    # Embedding model
    # embed_model = Ollama(model=Config.EMBED_MODEL_PATH)
    embed_model = HuggingFaceEmbedding(
        model_name=CONFIG.EMBED_MODEL_PATH
        # encode_kwargs={
        #     'normalize_embeddings': True,
        #     'device': 'cuda' if hasattr(Settings, 'device') else 'cpu'
        # }
    )

    # LLM
    llm_model = Ollama(model=CONFIG.LLM_MODEL_PATH)

    # Global Configuration
    Settings.embed_model = embed_model
    Settings.llm = llm_model

    # Verify Embedding model
    test_embedding = embed_model.get_text_embedding("测试文本")
    print(f"Embedding Dimension Verification： {len(test_embedding)}")

    prompt = "Hello, how are you?"
    output = llm_model.complete(prompt=prompt)
    print(output)

    return embed_model, llm_model

############Data Processing###############
"""
THis is RAG for Law assistant, the requirement for local data as following:
1. Law item need be clear, each item will be isolated. 
2. Each Law should have unique source 

Each law should be used as one node in vector database.
"""
def load_and_validate_json_files(data_dir:str) -> List[Dict]:
    # Load and Verify Json format law document
    json_files = list(Path(data_dir).glob("*.json"))
    assert json_files, f"Can not find JSON in {data_dir}"

    all_data = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # Verify data structure
                if not isinstance(data, list):
                    raise ValueError(f"File{json_file.name}root element should be list")
                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError(f"File{json_file.name}includes non-dictionary element")
                    for k, v in item.items():
                        if not isinstance(v, str):
                            raise ValueError(f"File{json_file.name} Key'{k}'value is not a string")
                    all_data.extend(
                        {
                            "context": item,
                            "metadata": {"source:": json_file.name}
                        }for item in data
                    )

            except Exception as e:
                raise RuntimeError(f"Load file{json_file}fail： {str(e)}")
    print(f"Load{len(all_data)}law items successfully")
    return all_data

# Create node
# TextNode is simple and use string


def create_nodes(raw_data: List[Dict]) -> List[TextNode]:
    "Add ID to ensure stable"
    nodes = []
    for entry in raw_data:
        law_dict = entry["context"]
        source_file = entry["metadata"]["source:"]

        for full_title, content in law_dict.items():
            # Generate Stable ID(Avoid repeated)
            node_id = f"{source_file}::{full_title}"

            parts = full_title.split(" ", 1)
            law_name = parts[0] if len(parts) > 0 else "unknown law"
            article = parts[1] if len(parts) > 1 else "unknown item"

            node = TextNode(
                text=content,
                id_=node_id,  # setup stable ID explicitly
                metadata={
                    "law_name": law_name,
                    "article": article,
                    "full_title": full_title,
                    "source_file": source_file,
                    "content_type": "legal_article"
                }
            )
            nodes.append(node)

    print(f"Generate {len(nodes)} text node （ID Example：{nodes[0].id_}）")
    return nodes

# ================== Vector Storing==================

def init_vector_store(nodes: List[TextNode]) -> VectorStoreIndex:
    print(CONFIG.VECTOR_DB_PATH)
    chroma_client = chromadb.PersistentClient(path=CONFIG.VECTOR_DB_PATH)
    chroma_collection = chroma_client.get_or_create_collection(
        name=CONFIG.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Make sure storage context initialization correctly
    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
    )

    # Whether need create new index
    if chroma_collection.count() == 0 and nodes is not None:
        print(f"Create new index（{len(nodes)} nodes）...")

        # Add node to storage context explicitly.
        storage_context.docstore.add_documents(nodes)

        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True
        )
        # double persistent to store
        storage_context.persist(persist_dir=CONFIG.PERSIST_DIR)
        index.storage_context.persist(persist_dir=CONFIG.PERSIST_DIR)  # <-- Add
    else:
        print("Load existing index...")
        storage_context = StorageContext.from_defaults(
            persist_dir=CONFIG.PERSIST_DIR,
            vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
        )
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )

    # Verify security
    print("\nStorage verification result：")
    doc_count = len(storage_context.docstore.docs)
    print(f"DocStore records：{doc_count}")

    if doc_count > 0:
        sample_key = next(iter(storage_context.docstore.docs.keys()))
        print(f"Example Node ID：{sample_key}")
    else:
        print("Warning: Document storage is empty, please check the logic to add nodes！")

    return index

############RAG Encapsulation###############
class RAGSystem:
    def __init__(self):
        self.embed_model, self.llm = init_models()
        self.index = self._init_index()
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=CONFIG.TOP_K,
            text_qa_template=response_template,
            verbose=True
        )

    def _init_index(self):
        if not Path(CONFIG.VECTOR_DB_PATH).exists():
            print("\nInitialize Data...")
            raw_data = load_and_validate_json_files(CONFIG.DATA_DIR)
            nodes = create_nodes(raw_data)
        else:
            nodes = None

        return init_vector_store(nodes)

    def query(self, question: str) -> str:
        response = self.query_engine.query(question)
        return response.response

    async def query_streaming(self, question: str) -> AsyncGenerator[str, None]:
        # 启用流式响应
        self.query_engine = self.index.as_query_engine(
            streaming=True,  # 关键设置
            similarity_top_k=3
        )

        response = await self.query_engine.aquery(question)  # 异步查询
        async for chunk in response.response_gen:  # 流式获取
            yield chunk


############Main###############
def main():
    embed_model, llm = init_models()

    # only execute when need update data
    if not Path(CONFIG.VECTOR_DB_PATH).exists():
        print("\nInitialize Data...")
        raw_data = load_and_validate_json_files(CONFIG.DATA_DIR)
        nodes = create_nodes(raw_data)
    else:
        nodes = None  # Not loading when have existing data

    print("\nInitialize vector storage...")
    start_time = time.time()
    index = init_vector_store(nodes)

    # From local documents to test
    # documents = SimpleDirectoryReader("data").load_data()
    # print(documents)
    # index = VectorStoreIndex.from_documents(documents)

    print(f"Time to load index：{time.time() - start_time:.2f}s")
    print(index)

    # Create search engine
    query_engine = index.as_query_engine(
        similarity_top_k=CONFIG.TOP_K,
        text_qa_template=response_template,
        verbose=True
    )
    # query_engine = index.as_chat_engine(
    #     similarity_top_k=CONFIG.TOP_K,
    #     text_qa_template=response_template,
    #     verbose=True
    # )

    # Example of search
    while True:
        question = input("\n请输入消费者保护法相关问题（Input 'q' to quit）: ")
        if question.lower() == 'q':
            break

        # execute search
        response = query_engine.query(question)
        # response = query_engine.chat(question)

        # Present result
        print(f"\nAssistant Answer：\n{response.response}")
        print("\nSupporting Evidence：")
        for idx, node in enumerate(response.source_nodes, 1):
            meta = node.metadata
            print(meta)
            print(f"\n[{idx}] {meta['full_title']}")
            print(f"  source：{meta['source_file']}")
            print(f"  name of law：{meta['law_name']}")
            print(f"  content of item：{node.text[:]}")
            print(f"  Similarity score：{node.score:.4f}")

def get_rag_system():
    return RAGSystem()

if __name__ == "__main__":
    rag = RAGSystem()
    # main()