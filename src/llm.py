'''
===========================================
        Module: Open-source LLM Setup
===========================================
'''
from langchain.llms import CTransformers
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from dotenv import find_dotenv, load_dotenv
import box
import yaml

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


# Dummy model to do nothing
class SearchOnlyLLM(LLM):
    response = 'only search documents'

    @property
    def _llm_type(self) -> str:
        return "search-only-llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self.response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"responses": [self.response]}


def build_llm():
    if cfg.SEARCH_ONLY:
        llm = SearchOnlyLLM()
    else:
        # Local CTransformers model
        llm = CTransformers(model=cfg.MODEL_BIN_PATH,
                         model_type=cfg.MODEL_TYPE,
                         config={'max_new_tokens': cfg.MAX_NEW_TOKENS,
                                 'temperature': cfg.TEMPERATURE}
                         )

    return llm
