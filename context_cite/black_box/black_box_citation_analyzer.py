"""
# 黑盒验证研究

设有下面场景。

1. 设有用户的原始提问 `query`；
2. 且有一份目标参考资料 `target_text`;
3. 用户使用指定模型 `M`，通过将原始提问 `query` 直接发给模型服务，在模型服务内部关联到一份外部无法得知的上下文（即不确定模型收到的最终INPUT是否包含前述目标参考资料 `target_text`），从而得到了输出 `output_1`;
4. 用户使用指定模型 `M`，通过将原始提问 `query` 结合目标参考资料 `target_text` 后发给模型服务，在模型服务内部关联到一份外部无法得知的上下文（但模型收到的最终INPUT必然包含前述目标参考资料 `target_text`），从而得到了输出 `output_2`;
5. 用户本地可以提供与指定模型 `M` 同架构但参数规模和量化上有差异的模型 `m`。

基于现有的 context_cite 的方法论和实现，以本地模型`m`拟合`M`，通过下面的可知变量（`query`、`target_text`、`output_1`、`output_2`），得到 `target_text` 对 `output_1` 的影响程度的量化结论。

"""
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..context_citer import DEFAULT_PROMPT_TEMPLATE
from ..solver import LassoRegression
from ..utils import aggregate_logit_probs, get_masks_and_logit_probs


class _SingleBlockPartitioner:
    """
    单块 partitioner：将整段 context 视为一个 source，用于 BlackBox 归因。
    不暴露到主包，仅本模块使用。
    """

    def __init__(self, context: str) -> None:
        self.context = context

    @property
    def num_sources(self) -> int:
        return 1

    def split_context(self) -> None:
        pass

    def get_source(self, index: int) -> str:
        if index != 0:
            raise IndexError(index)
        return self.context

    def get_context(self, mask: Optional[NDArray] = None) -> str:
        if mask is None or (len(mask) > 0 and mask[0]):
            return self.context
        return ""


class BlackBoxCitationResult:
    """BlackBox 归因结果：target_text 对 output_1、output_2 的归因分数。"""

    def __init__(
        self,
        influence_on_output_1: float,
        influence_on_output_2: float,
    ) -> None:
        self.__influence_on_output_1 = influence_on_output_1
        self.__influence_on_output_2 = influence_on_output_2

    def get_influence_on_output_1(self) -> float:
        return self.__influence_on_output_1

    def get_influence_on_output_2(self) -> float:
        return self.__influence_on_output_2


class BlackBoxCitationAnalyzer:
    def __init__(self, model_source: str) -> None:
        self.__model_source = model_source
        self.__model = None
        self.__tokenizer = None

    def _get_model_and_tokenizer(self):
        if self.__model is None or self.__tokenizer is None:
            self.__model = AutoModelForCausalLM.from_pretrained(self.__model_source)
            self.__model.to("cuda")
            self.__tokenizer = AutoTokenizer.from_pretrained(self.__model_source)
            self.__tokenizer.padding_side = "left"
            if self.__tokenizer.pad_token is None:
                self.__tokenizer.pad_token = self.__tokenizer.eos_token
        return self.__model, self.__tokenizer

    def analyze(
        self,
        query: str,
        target_text: str,
        output_1: str,
        output_2: str,
    ) -> BlackBoxCitationResult:
        model, tokenizer = self._get_model_and_tokenizer()
        partitioner = _SingleBlockPartitioner(context=target_text)
        solver = LassoRegression()
        num_ablations = 64
        ablation_keep_prob = 0.5
        batch_size = 1
        prompt_template = DEFAULT_PROMPT_TEMPLATE

        def get_prompt_ids(mask: Optional[NDArray] = None):
            context = partitioner.get_context(mask)
            prompt = prompt_template.format(context=context, query=query)
            messages = [{"role": "user", "content": prompt}]
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return tokenizer.encode(chat_prompt, add_special_tokens=False)

        def _attribution_for_response(response_text: str) -> float:
            response_ids = tokenizer.encode(
                response_text, add_special_tokens=False
            )
            if len(response_ids) == 0:
                return 0.0
            masks, logit_probs = get_masks_and_logit_probs(
                model,
                tokenizer,
                num_ablations,
                partitioner.num_sources,
                get_prompt_ids,
                response_ids,
                ablation_keep_prob,
                batch_size,
                base_seed=0,
            )
            outputs = aggregate_logit_probs(logit_probs, output_type="logit_prob")
            num_output_tokens = len(response_ids)
            weight, _ = solver.fit(masks, outputs, num_output_tokens)
            return float(weight[0])

        influence_on_output_1 = _attribution_for_response(output_1)
        influence_on_output_2 = _attribution_for_response(output_2)
        return BlackBoxCitationResult(
            influence_on_output_1=influence_on_output_1,
            influence_on_output_2=influence_on_output_2,
        )