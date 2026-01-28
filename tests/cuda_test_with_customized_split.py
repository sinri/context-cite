# todo: use macos local model to test with customized separator-based splitting

import re
from context_cite.context_citer import ContextCiter
from context_cite.context_partitioner import SeparatorContextPartitioner

# model_dir='/Users/sinri/code/huggineface/Mistral-7B-v0.3'
# model_dir = '/Users/sinri/code/huggineface/Dolphin3.0-Llama3.2-1B'
model_dir = "E:\\sinri\\DeepSeek-R1-Distill-Qwen-1.5B"

# Context with custom separators (using ==== as standalone line markers)

with open('../data/long_context.md', 'r') as file:
    context = file.read()

query = "how long did Noah stay in the ark?"

# Create a custom partitioner using separator-based splitting
partitioner = SeparatorContextPartitioner(
    context=context,
    separator=re.compile(r"^[ -]+$"),
    match_whole_line=True,  # Match standalone lines with ====
    strip_sources=True  # Strip whitespace from each source
)

print(f"Number of sources: {partitioner.num_sources}")
print("\nSources:")
for i, source in enumerate(partitioner.sources):
    print(f"\nSource {i} (first 100 chars): {source[:100]}...")

# Create ContextCiter with the custom partitioner
cc = ContextCiter.from_pretrained(
    model_dir,
    context,
    query,
    partitioner=partitioner  # Use the custom separator-based partitioner
)

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Response:")
    print("=" * 80)
    print(cc.response)
    
    print("\n" + "=" * 80)
    print("Attributions (Top 8):")
    print("=" * 80)
    data = cc.get_attributions(as_dataframe=True, top_k=8, verbose=False).data
    print(data)
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"Total sources: {cc.num_sources}")
    print(f"Query: {query}")
    print(f"Response length: {len(cc.response)} characters")
