import torch
import transformers
from transformers import AutoTokenizer

from torch import nn as nn

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from lm_eval.models.huggingface import HFLM
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

# Custom implementation since stop_sequences_criteria is not available
class StopSequenceCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_sequences, input_len, batch_size):
        self.tokenizer = tokenizer
        self.stop_sequences_ids = [
            tokenizer.encode(seq, add_special_tokens=False, return_tensors="pt")[0]
            for seq in stop_sequences
        ]
        self.input_len = input_len
        self.batch_size = batch_size
    
    def __call__(self, input_ids, scores, **kwargs):
        # Only look at the generated text after self.input_len tokens
        # For each sequence in the batch
        for batch_idx in range(min(self.batch_size, input_ids.shape[0])):
            sequence = input_ids[batch_idx, self.input_len:]
            for stop_sequence in self.stop_sequences_ids:
                # Check if the sequence ends with any of the stop sequences
                if len(sequence) >= len(stop_sequence) and torch.all(
                    sequence[-len(stop_sequence):] == stop_sequence
                ):
                    return True
        return False

def stop_sequences_criteria(tokenizer, stop_sequences, input_len, batch_size):
    if not stop_sequences:
        return StoppingCriteriaList()
    return StoppingCriteriaList([StopSequenceCriteria(tokenizer, stop_sequences, input_len, batch_size)])

@register_model("mamba_hybrid")
class MambaEvalWrapper(HFLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained, max_length=2048, batch_size=None, device="cuda",
                 dtype=torch.bfloat16):
        LM.__init__(self)
        from mamba.hybrid_wrapper import MambaTransformerHybridModelWrapper
        self._model = MambaTransformerHybridModelWrapper.from_pretrained(pretrained, torch_dtype=dtype).model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        print(self._model)
        self._model = self._model.cuda()
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

    # this is copied from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py#L824-L849
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
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        return self._model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=False,
            **generation_kwargs,
        )

@register_model("mamba2_hybrid")
class Mamba2EvalWrapper(HFLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained, max_length=2048, batch_size=None, device="cuda",
                 dtype=torch.bfloat16):
        LM.__init__(self)
        from mamba2.hybrid_wrapper import MambaTransformerHybridModelWrapper
        self._model = MambaTransformerHybridModelWrapper.from_pretrained(pretrained, torch_dtype=dtype).model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        print(self._model)
        self._model = self._model.cuda()
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

    # this is copied from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py#L824-L849
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
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        return self._model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=False,
            **generation_kwargs,
        )
        
if __name__ == "__main__":
    cli_evaluate()
