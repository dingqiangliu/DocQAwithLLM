# =========================
#  Module: Vector DB Build
# =========================
import box
import yaml
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from src.idol import IDOL

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


# Build vector database
def run_db_build():
    loaders = [DirectoryLoader(cfg.DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader),
               DirectoryLoader(cfg.DATA_PATH,
                                        glob='*.docx',
                                        loader_cls=Docx2txtLoader)]
    documents = [doc for ld in loaders for doc in ld.load()]

    if cfg.REG_SEPARATORS:
        text_splitter = CharacterTextSplitter(separator=cfg.REG_SEPARATORS,
                                              is_separator_regex=True,
                                              keep_separator=True,
                                              chunk_size=cfg.CHUNK_SIZE,
                                              chunk_overlap=cfg.CHUNK_OVERLAP)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                       chunk_overlap=cfg.CHUNK_OVERLAP)

    if not documents:
        documents = [Document(page_content='unkown', metadata={'source': 'unkown'})]

    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDINGS_MODEL,
                                       model_kwargs={'device': cfg.DEVICE})

    if cfg.VECTOR_DB == 'IDOL' :
        IDOL.from_documents(texts, embeddings, url = cfg.IDOL_INDEX_URL,
                            field = cfg.IDOL_VECTOR_FIELD,
                            database = cfg.IDOL_DATABASE,
                            index_batch_size = cfg.IDOL_INDEX_BATCH_SIZE,
                            vector_search = cfg.IDOL_VECTOR_SEARCH)
    else:
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(cfg.DB_FAISS_PATH)


if __name__ == "__main__":
    run_db_build()
