import os
import time
from datetime import datetime

from context_cite.black_box import BlackBoxCitationAnalyzer
from tests.blackbox.utils import LLM, ReportCsvHelper


def get_current_time_str():
    """
    获取当前时间的字符串，格式为 YmdHis（年月日时分秒）

    Returns:
        str: 格式化后的时间字符串，例如 "20260203102030"
    """
    # 获取当前本地时间
    current_time = datetime.now()
    # 按照 YmdHis 格式格式化时间
    time_str = current_time.strftime("%Y%m%d%H%M%S")
    return time_str


class LocalExpExecutor:
    def __init__(self, analyzer: BlackBoxCitationAnalyzer, exp_dir: str):
        self.__analyzer = analyzer
        self.__llm = LLM(analyzer)
        self.__exp_dir = exp_dir
        self.__query = None
        self.__target_text = None
        time.time()
        self.__reporter = ReportCsvHelper(os.path.join(self.__exp_dir, "report-" + get_current_time_str() + ".csv"))

    def read_query(self) -> str:
        if self.__query is None:
            p = os.path.join(self.__exp_dir, "query.txt")
            with open(p, "r", encoding="utf-8") as f:
                self.__query = f.read()
        return self.__query

    def read_target_text(self) -> str:
        if self.__target_text is None:
            p = os.path.join(self.__exp_dir, "target.txt")
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
        return self.__target_text

    def __generate_prompt(self, query: str, target_text: str = ''):
        return f"# 可能有用的参考信息：\n\n{target_text}\n\n# 需要回应的问题：\n\n{query}"

    def execute_single_round(self):
        query = self.read_query()
        target_text = self.read_target_text()

        prompt_1 = self.__generate_prompt(query)
        output_1 = self.__llm.generate(prompt_1)

        prompt_2 = self.__generate_prompt(query, target_text)
        output_2 = self.__llm.generate(prompt_2)

        result = self.__analyzer.analyze(query, target_text, output_1, output_2)
        self.__reporter.add_record(prompt_1, prompt_2, output_1, output_2,
                                   result.get_influence_on_output_1(),
                                   result.get_influence_on_output_2()
                                   )

    def execute(self, rounds: int):
        for i in range(rounds):
            self.execute_single_round()
        self.__reporter.save()
