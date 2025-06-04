import torch

import transformers
from transformers import AutoTokenizer

from torch import nn as nn

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from lm_eval.models.huggingface import HFLM
import torch

@register_model("mamba_hybrid")
class MambaEvalWrapper(HFLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained, max_length=2048, batch_size=None, device="cuda",
                 dtype=torch.bfloat16):
        LM.__init__(self)
        from mamba.hybrid_wrapper import MambaTransformerHybridModelWrapper
        
        # Use the wrapper directly
        self._model = MambaTransformerHybridModelWrapper.from_pretrained(pretrained, torch_dtype=dtype)
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        print(self._model)
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        
        # Required HFLM attributes
        self.backend = "causal"
        self.add_bos_token = False
        self.logits_cache = None
        self.revision = None
        self.truncation = False
        self.trust_remote_code = True
        self.use_fast_tokenizer = True
        self.torch_dtype = dtype
        self.low_cpu_mem_usage = None
        self.device_map = None
        self.model_parallel = False
        self.device_map_option = None
        self.parallelize = False
        self.padding_side = "right"
        self.model_name = pretrained
        
        self._device = torch.device(device)

    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def max_length(self):
        return self._max_length
    
    def get_model_info(self):
        """
        Returns information about the model
        """
        return {
            "model_name": self.model_name,
            "model_revision": self.revision,
            "batch_size": self.batch_size,
            "device": str(self._device),
        }

    # this is copied from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py#L896-L921
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        
        # mamba_ssm GenerationMixin does not support stopping_criteria, so we generate to max_length and truncate
        return self._model.generate(
            input_ids=context,
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            **generation_kwargs,
        )

@register_model("mamba2_hybrid")
class Mamba2EvalWrapper(HFLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained, max_length=2048, batch_size=None, device="cuda",
                 dtype=torch.bfloat16):
        LM.__init__(self)
        from mamba2.hybrid_wrapper import MambaTransformerHybridModelWrapper
        
        # Load the wrapper - note it returns a wrapper, not the model directly
        wrapper = MambaTransformerHybridModelWrapper.from_pretrained(pretrained, torch_dtype=dtype)
        self._model = wrapper  # Use the wrapper, not wrapper.model
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        print(self._model)
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length

        # Required HFLM attributes
        self.backend = "causal"
        self.add_bos_token = False
        self.logits_cache = None
        self.revision = None
        self.truncation = False
        self.trust_remote_code = True
        self.use_fast_tokenizer = True
        self.torch_dtype = dtype
        self.low_cpu_mem_usage = None
        self.device_map = None
        self.model_parallel = False
        self.device_map_option = None
        self.parallelize = False
        self.padding_side = "right"
        self.model_name = pretrained
        
        self._device = torch.device(device)

    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def max_length(self):
        return self._max_length
    
    def get_model_info(self):
        """
        Returns information about the model
        """
        return {
            "model_name": self.model_name,
            "model_revision": self.revision,
            "batch_size": self.batch_size,
            "device": str(self._device),
        }

    # this is copied from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py#L896-L921
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        
        # mamba_ssm GenerationMixin does not support stopping_criteria, so we generate to max_length and truncate
        return self._model.generate(
            input_ids=context,
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            **generation_kwargs,
        )
        
if __name__ == "__main__":
    cli_evaluate()