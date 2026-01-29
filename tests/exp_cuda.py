import csv
import json
import os
import random
import re

from context_cite import SeparatorContextPartitioner
from context_cite.citation_analyzer import CitationAnalyzer


class Exp:
    def __init__(self,model_source:str):
        self.__model_source=model_source


    def __execute(self,context:str,query:str):
        partitioner = SeparatorContextPartitioner(
            context=context,
            separator=re.compile(r"^----$"),
            match_whole_line=True,  # Match standalone lines with ====
            strip_sources=True  # Strip whitespace from each source
        )
        result = CitationAnalyzer(model_source=self.__model_source).execute(
            context=context,
            query=query,
            top_k=3,
            verbose=False,
            max_new_tokens=2048,
            do_sample=False,
            partitioner=partitioner,
        )
        print(f"[RESPONSE]\n {result.get_response()}")

        print(f"[ITEMS]")
        for datum in result.items():
            print(f"[DATUM] index: {datum.get_index()}, score: {datum.get_score()}, source: {datum.get_source_head()}")

        print(f"[NORMALIZED ITEMS]")
        result.normalize()
        for datum in result.items():
            print(f"[DATUM] index: {datum.get_index()}, score: {datum.get_score()}, source: {datum.get_source_head()}")

        print(f"[FILTERED ITEMS]")
        result.filter(lowest_score=0.5)
        for datum in result.items():
            print(f"[DATUM] index: {datum.get_index()}, score: {datum.get_score()}, source: {datum.get_source_head()}")

        return result

    def perform(self,case_code:str, repeat:int):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        case_dir = os.path.join(base_dir, "data", "internal", "t", case_code)

        blocks=[]
        for i in range(8):
            t_i_file = os.path.join(case_dir, f"t_{i}.txt")
            with open(t_i_file,"r",encoding="utf-8") as f:
                block=f"Reference [{i}]\n\n"+("\n".join(f.readlines()))
                blocks.append(block)

        queries_file = os.path.join(case_dir, "queries.txt")
        with open(queries_file, "r", encoding="utf-8") as f:
            queries = f.readlines()

        reports = []
        for round1 in range(repeat):
            print(f"Round: {round1}")

            random.shuffle(blocks)

            context="\n\n----\n\n".join(blocks)

            print("context: ", context)

            if len(queries)>0:
                report_of_one_repeat={}
                for query in queries:
                    print(f"handle query: {query}")
                    report_of_one_repeat[query]=[]
                    for round2 in range(repeat):
                        print(f"Repeat on query: {round2}")
                        result=self.__execute(context,query)
                        report_atomic=[]
                        for item in result.items():
                            report_atomic.append(item.to_dict())
                        report_of_one_repeat[query].append(report_atomic)
                reports.append(report_of_one_repeat)

        report_file=os.path.join(case_dir, "report.json")
        with open(report_file,"w",encoding="utf-8") as f:
            f.write(json.dumps(reports,indent=2,ensure_ascii=False))
            print(f"written report: {report_file}" )


if __name__ == '__main__':
    model_source = "E:\\sinri\\DeepSeek-R1-Distill-Qwen-1.5B"
    exp=Exp(model_source)
    exp.perform("d",repeat=5)
