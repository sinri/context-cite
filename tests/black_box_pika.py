from context_cite.citation_analyzer import CitationAnalyzer
model_source = "E:\\sinri\\DeepSeek-R1-Distill-Qwen-1.5B"
context="""
2027年初，将会有一批备受期待的网络游戏集中上线开玩。
怀旧党可以玩到《春之谷》、《雷霆军阵》等重制复刻版本。
对于非氪金用户，动作游戏《勇气之杖》可以提供免费的体验。
另外，《野蛮X》可能于2026年底就发布预览版，可以一试。
主机游戏方面则动作不多，仅《沉睡的阿卡多的宝藏》系列有相关预告，其余游戏基本都是跳票。
《沉睡的阿卡多的宝藏》系列将于2027年2月发布《荒野之章》和《海洋宝藏》两个DLC。
"""
query="""
主机游戏《沉睡的阿卡多的宝藏——荒野之章》什么时候发售啊？
"""
if __name__ == "__main__":
    result = CitationAnalyzer(model_source=model_source).execute(
        context=context,
        query=query,
        top_k=10,
        verbose=False,
        max_new_tokens=2048,
        do_sample=False,
        # partitioner=partitioner,
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
