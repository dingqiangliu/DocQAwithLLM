# =========================
#  Module: ChartGLM model
# =========================
import box
import yaml
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from typing import Any, Dict, List, Mapping, Optional
from transformers import AutoTokenizer, AutoModel

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


class ChatGLM(LLM):
    """ models for THUDM/chatglm-6b or THUDM/chatglm2-6b
    """
    
    chatglm_model: Any  #: :meta private:
    chatglm_tokenizer: Any  #: :meta private:
        
    model: str = None
    config: Optional[Dict[str, Any]] = None
       
    def __init__(self, **kwargs: Any) -> LLM:
        """
        Parameters
        ----------
        model : str
            The name of the model file in repo or directory.

        Returns
        -------
        LLM
            model hold ChatGLM.

        """
        super().__init__(**kwargs)
        
        self.chatglm_model = AutoModel.from_pretrained(self.model,
                                                       trust_remote_code=True,
                                                       device=cfg.DEVICE).float()
        self.chatglm_model = self.chatglm_model.eval()
        self.chatglm_tokenizer = AutoTokenizer.from_pretrained(self.model, 
                                                               trust_remote_code=True,
                                                               device=cfg.DEVICE)

                                          
    @property
    def _llm_type(self) -> str:
        return "chatglm-llm"


    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        response, history = self.chatglm_model.chat(self.chatglm_tokenizer, 
                                                    prompt, 
                                                    **self.config)
        return response


    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            'model': self.model,
            'config': self.config
        }
