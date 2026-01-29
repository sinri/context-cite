import json
from typing import List, Optional

from context_cite import ContextCiter
from pandas.core.frame import DataFrame

from context_cite.context_partitioner import BaseContextPartitioner


class CitationAnalyzedDatum:
    def __init__(self, index: int, score: float, source: str):
        self.__index = index
        self.__score = score
        self.__source = source

    def get_index(self):
        return self.__index

    def get_score(self):
        return self.__score

    def get_source(self):
        return self.__source

    def set_score(self, score: float):
        self.__score = score

    def get_source_head(self):
        x = self.__source.find("\n")
        return self.__source[0:x]

    def to_dict(self):
        return {"index": self.__index, "score": self.__score, "source_head":self.get_source_head()}


class CitationAnalyzedData:
    def __init__(self, datum_list: List[CitationAnalyzedDatum], response: str):
        self.__datum_list = datum_list
        self.__response = response

    def normalize(self):
        """
        将 self.__datum_list 的 item 的 score 归一化处理（即最高分的项的分数定义为1.00）。
        """
        if not self.__datum_list:
            return

        max_score = max(datum.get_score() for datum in self.__datum_list)
        if max_score > 0:
            for datum in self.__datum_list:
                datum.set_score(datum.get_score() / max_score)

    def filter(self, lowest_score: float):
        """
        去掉 score 低于 lowest_score 的元素。
        """
        self.__datum_list = [datum for datum in self.__datum_list if datum.get_score() >= lowest_score]

    def items(self) -> List[CitationAnalyzedDatum]:
        return self.__datum_list

    def get_response(self):
        return self.__response


class CitationAnalyzer:
    def __init__(self, model_source: str):
        self.__model_source = model_source

    def execute(
            self,
            context: str,
            query: str,
            top_k: int = 5,
            verbose: bool = False,
            max_new_tokens: int = 1024,
            do_sample: bool = False,
            partitioner: Optional[BaseContextPartitioner] = None,
    ) -> CitationAnalyzedData:
        cc = ContextCiter.from_pretrained(
            self.__model_source,
            context=context,
            query=query,
            generate_kwargs={"max_new_tokens": max_new_tokens, "do_sample": do_sample},
            partitioner=partitioner,
        )
        data = cc.get_attributions(as_dataframe=True, top_k=top_k, verbose=verbose).data

        result_list = []
        for index, row in data.iterrows():
            datum = CitationAnalyzedDatum(
                index=int(index),
                score=float(row['Score']),
                source=str(row['Source'])
            )
            result_list.append(datum)

        return CitationAnalyzedData(result_list, cc.response)
