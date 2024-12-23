'''
===========================================
        Module: Util functions
===========================================
'''
import box
import yaml

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from src.prompts import qa_template
from src.llm import build_llm
from src.idol import IDOL

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def set_qa_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])
    return prompt


def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT}),
                                       return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return dbqa


def setup_dbqa():
    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDINGS_MODEL,
                                       model_kwargs={'device': cfg.DEVICE})

    if cfg.VECTOR_DB == 'IDOL' :
        vectordb = IDOL(embeddings, url = cfg.IDOL_SEARCH_URL,
                        vector_field = cfg.IDOL_VECTOR_FIELD,
                        database = cfg.IDOL_DATABASE,
                        search_type = cfg.IDOL_SEARCH_TYPE)
    else:
        vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)

    llm = build_llm()
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa
