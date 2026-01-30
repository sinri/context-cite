from typing import List, Dict

from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import pipeline
import os

class LLM:
    def __init__(self, model_source):
        self.__pipe = pipeline("text-generation", model=model_source)

    def generate(self, messages:List[Dict[str,str]], remove_think=True, repetition_penalty=1.1):
        response = self.__pipe(messages, max_new_tokens=10240, repetition_penalty=repetition_penalty)
        res=response[0]["generated_text"][1]["content"]
        if remove_think:
            if "</think>" in res:
                res = res.split("</think>")[-1]
            res = res.lstrip()
        return res

class ExpCaseBuilder:
    def __init__(self, llm: LLM):
        self.__llm = llm

    def build(self,case_code:str):
        # Use robust path relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        case_dir = os.path.join(base_dir, "data", "internal", "t", case_code)
        
        t0_file = os.path.join(case_dir, "t0.txt")
        with open(t0_file,"r",encoding="utf-8") as f:
            t0_content=f.read()
            self.__generate(case_dir,t0_content,True,True,True)
            self.__generate(case_dir,t0_content,True,True,False)
            self.__generate(case_dir,t0_content,True,False,True)
            self.__generate(case_dir,t0_content,True,False,False)
            self.__generate(case_dir,t0_content,False,True,True)
            self.__generate(case_dir,t0_content,False,True,False)
            self.__generate(case_dir,t0_content,False,False,True)
            self.__generate(case_dir,t0_content,False,False,False)

    def __generate(self,case_dir,t0_content,formatted:bool,normalized:bool,contexted:bool):
        n1=0b001 if formatted else 0b000
        n2=0b010 if normalized else 0b000
        n3=0b100 if contexted else 0b000
        n=n1|n2|n3
        fn=f"t_{n}"

        query="将下面给出的原始文本内容，在内容不变的情况下，按照下面的要求按需调整其文本特征并直接输出。\n\n"
        if formatted:
            query+="要求1：确保文本格式为Markdown，顶级标题级别为`##`；\n\n"
        else:
            query+="要求1：确保文本为非markdown格式的自然行文；\n\n"
        if normalized:
            query+="要求2：确保内容以标准的书面语表示，保证用词的正规性和语法的正确性；\n\n"
        else:
            query+="要求2：确保内容以口语化、日常化的表达，避免标准化公文化；\n\n"
        if contexted:
            query+="要求3：确保文中没有缺失上下文的内容，对于需要解释的特定词汇补充必要的解释；\n\n"
        else:
            query+="要求3：消除文中可能存在冗余的常见事物解释，降低冗余；\n\n"
        query+=f"原始文本内容：\n\n{t0_content}"

        messages = [
            {"role": "user", "content": query},
        ]
        res=self.__llm.generate(messages)

        output_file = os.path.join(case_dir, f"{fn}.txt")
        with open(output_file,"w",encoding="utf-8") as f:
            f.write(res)
            print(f'built version of as {"markdown" if formatted else "plain"}, {"normalized" if normalized else "casual"}, {"contexted" if contexted else "brief"} to {fn}')


if __name__ == "__main__":
    model_source="E:\\sinri\\DeepSeek-R1-Distill-Qwen-1.5B"

    llm = LLM(model_source)
    builder=ExpCaseBuilder(llm)
    builder.build("a")
    # builder.build("b")
    # builder.build("c")
    # builder.build("d")
