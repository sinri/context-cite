from typing import List, Dict

from transformers import pipeline

from context_cite.black_box import BlackBoxCitationAnalyzer


class LLM:
    def __init__(self, model_source):
        self.__pipe = pipeline("text-generation", model=model_source)

    def generate(self, messages:List[Dict[str,str]], remove_think=True):
        response = self.__pipe(messages, max_new_tokens=10240, repetition_penalty=1.5)
        res=response[0]["generated_text"][1]["content"]
        if remove_think:
            if "</think>" in res:
                res = res.split("</think>")[-1]
            res = res.lstrip()
        return res

if __name__ == '__main__':
    # model_source="E:\\sinri\\DeepSeek-R1-Distill-Qwen-1.5B"
    model_source="E:\\sinri\\HuggingFace\\Qwen3-1.7B"

    query = "What is the capital of Bioland?"
    target_text = "Findois is the capital and largest city of Bioland."

    llm=LLM(model_source)
    output_1=llm.generate(messages=[{"role":"user","content":query}],) # repetition_penalty=1.5
    # output_1 = "I'm not sure."  # 无 target_text 时的典型短回答
    output_2 = llm.generate(messages=[{"role": "user", "content": target_text+"\n\n"+query}], ) # repetition_penalty=1.5
    # output_2 = "Findois is the capital of Bioland."  # 有 target_text 时的回答

    analyzer = BlackBoxCitationAnalyzer(model_source=model_source)
    result = analyzer.analyze_repeatedly(
        query=query,
        target_text=target_text,
        output_1=output_1,
        output_2=output_2,
        repeat=5,
    )

    influence_1 = result.get_influence_on_output_1()
    influence_2 = result.get_influence_on_output_2()

    # 数值越大，表示在本地模型 m 的视角下，这段回答越依赖 target_text；
    # 通常期望 influence_on_output_2 ≥ influence_on_output_1（因为 output_2 是在含 target_text 的上下文中生成的）。
    # > 0	target_text 作为 context 时，提高了模型生成该段 response 的“得分”（与“被引用”一致）
    # < 0	target_text 降低了该段 response 的得分（与引用相悖）
    # ≈ 0	几乎无影响

    print('output 1')
    print(output_1)
    print('output 2')
    print(output_2)
    print(influence_1, influence_2)