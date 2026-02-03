from context_cite.black_box import BlackBoxCitationAnalyzer
from tests.blackbox.local_exp_executor import LocalExpExecutor

# model_source="E:\\sinri\\DeepSeek-R1-Distill-Qwen-1.5B"
model_source = "E:\\sinri\\HuggingFace\\Qwen3-1.7B"

exp_dir = "/Users/sinri/code/context-cite/data/blackbox/exp1"


def perform():
    analyzer = BlackBoxCitationAnalyzer(model_source)
    executor = LocalExpExecutor(analyzer=analyzer, exp_dir=exp_dir)
    executor.execute(rounds=10)


if __name__ == '__main__':
    perform()
