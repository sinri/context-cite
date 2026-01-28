from context_cite.citation_analyzer import CitationAnalyzer


model_source="E:\\sinri\\DeepSeek-R1-Distill-Qwen-1.5B"

with open('../data/long_context.md', 'r') as file:
    context = file.read()

query = "how long did Noah stay in the ark?"

if __name__ == "__main__":
    result=CitationAnalyzer(model_source=model_source).execute(
        context=context,
        query=query,
        top_k=5,
        verbose=False,
        max_new_tokens=1024,
        do_sample=False,
    )
    print(f"[RESPONSE]\n {result.get_response()}")

    print(f"[ITEMS]")
    for datum in result.items():
        print(f"[DATUM] index: {datum.get_index()}, score: {datum.get_score()}, source: {datum.get_source()}")

    print(f"[NORMALIZED ITEMS]")
    result.normalize()
    for datum in result.items():
        print(f"[DATUM] index: {datum.get_index()}, score: {datum.get_score()}, source: {datum.get_source()}")

    print(f"[FILTERED ITEMS]")
    result.filter(lowest_score=0.5)
    for datum in result.items():
        print(f"[DATUM] index: {datum.get_index()}, score: {datum.get_score()}, source: {datum.get_source()}")