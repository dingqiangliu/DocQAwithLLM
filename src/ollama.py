# =========================
#  Module: model running by ollama
# =========================
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from typing import Any, Dict, List, Mapping, Optional
import ollama


class Ollama(LLM):
    """ models for ollama
    """
    
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

                                          
    @property
    def _llm_type(self) -> str:
        return "ollama-llm"


    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # TODO: pass self.config
        response = ollama.generate(self.model
                , prompt
                , options = self.config).response
        return response


    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            'model': self.model,
            'config': self.config
        }
