from transformers import GenerationConfig

from context_cite.black_box import BlackBoxCitationAnalyzer
from tests.blackbox.local_exp_executor import LocalExpExecutor

# model_source = "E:\\sinri\\DeepSeek-R1-Distill-Qwen-1.5B"

# model_source = "E:\\sinri\\HuggingFace\\Qwen3-1.7B"
# generation_config = GenerationConfig(
#     max_new_tokens=2048,
#     repetition_penalty=1.5,
#     temperature=0.7,
#     top_p=0.8,
#     top_k=20,
#     min_p=0,
# )

model_source = "E:\\sinri\\HuggingFace\\Qwen3-4B"
generation_config = GenerationConfig(
    max_new_tokens=2048,
    presence_penalty=1.5,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0,
)

exp_dir = "E:\\sinri\\context-cite\\data\\blackbox\\exp2"


def perform():
    analyzer = BlackBoxCitationAnalyzer(model_source)
    executor = LocalExpExecutor(analyzer=analyzer, generation_config=generation_config, exp_dir=exp_dir)
    executor.execute(rounds=10)


if __name__ == '__main__':
    perform()
