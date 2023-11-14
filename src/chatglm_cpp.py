# =========================
#  Module: model running by chatglm_cpp 
# =========================
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from typing import Any, Dict, List, Mapping, Optional
import chatglm_cpp


class ChatGLMCPP(LLM):
    """ models for THUDM/chatglm-6b or THUDM/chatglm2-6b
    """
    
    chatglm_model: Any  #: :meta private:
        
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
        
        self.chatglm_model = chatglm_cpp.Pipeline(model_path=self.model)

                                          
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
        response = self.chatglm_model.generate(prompt, **self.config)
        return response


    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            'model': self.model,
            'config': self.config
        }
