import re

from context_cite import SeparatorContextPartitioner
from context_cite.citation_analyzer import CitationAnalyzer


model_source="E:\\sinri\\DeepSeek-R1-Distill-Qwen-1.5B"

with open('../data/internal/sop/quality_sop.md', 'r',encoding='utf-8') as file:
    context = file.read()

query = "数据事故是怎么定义的?"

# Create a custom partitioner using separator-based splitting
partitioner = SeparatorContextPartitioner(
    context=context,
    separator=re.compile(r"^[ -]+$"),
    match_whole_line=True,  # Match standalone lines with ====
    strip_sources=True  # Strip whitespace from each source
)

if __name__ == "__main__":
    result=CitationAnalyzer(model_source=model_source).execute(
        context=context,
        query=query,
        top_k=5,
        verbose=False,
        max_new_tokens=1024,
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