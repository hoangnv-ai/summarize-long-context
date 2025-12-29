from ner import *
from semantic_chungking import *
from agent import Agent
from utils import *

with open("./raw_text/25112025_recorrect.txt",
          "r") as f:
    text = f.read()


# Văn bản mẫu (tin tức tiếng Việt)
sample_text = text

# Khởi tạo chunker
chunker = SemanticNewsChunker(
    similarity_threshold=0.50,
    min_chunk_size=200,
    max_chunk_size=500
)

# Thực hiện chunking-------------------------------
chunks = chunker.chunk(sample_text, verbose=True)
system_prompt = load_prompt()


previous_text = ""
for i_chunk, chunk in enumerate(chunks):
    agent = Agent(system=system_prompt)
    print(f"\n[Chunk {chunk['chunk_id']}] - {chunk['word_count']} từ, {chunk['sentence_count']} câu")
    print("-" * 80)
    if i_chunk == 0:
        chunk['previous_text'] = previous_text
        previous_text = chunk['text']
    else:
        chunk['previous_text'] = previous_text
        previous_text = chunk['text']

    # Thực hiện trích xuất entity của từng chunk-------------------------------
    list_entity_name = get_entity_name(chunk['text'])
    chunk["list_entity"] = list_entity_name
    print(list_entity_name)
    print("Bản tóm tắt :")
    # Thực hiện tóm tắt từng chunk-------------------------------
    result = agent(chunk)
    chunk["summarize"] = result

print(f"\n\nTổng số chunks: {len(chunks)}")

with open("./output/result.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)