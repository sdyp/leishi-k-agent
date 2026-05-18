from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
import json
import pickle
import os

embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key="sk-7174ddfc6635444fa4319c9732295f99"
)

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_texts(data, text_field):
    texts = []
    for item in data:
        if text_field in item:
            texts.append(item[text_field])
    return texts

# 确保目录存在
os.makedirs("faiss_indexes", exist_ok=True)

# ========== 1. 构建 FAISS 向量库 ==========
# 设备库
device_data = load_json_data("data/device_manual.json")
device_texts = extract_texts(device_data, "content")
device_kb = FAISS.from_texts(device_texts, embeddings)
device_kb.save_local("faiss_indexes/device_kb")

# 销售库
sales_data = load_json_data("data/sales_channel.json")
sales_texts = extract_texts(sales_data, "content")
sales_kb = FAISS.from_texts(sales_texts, embeddings)
sales_kb.save_local("faiss_indexes/sales_kb")

# 歌曲库（FAISS）
song_data = load_json_data("data/songs.json")
song_texts = []
for item in song_data:
    combined = f"{item['title']} {item['artist']} {item['genre']} {item['description']}"
    song_texts.append(combined)
song_kb = FAISS.from_texts(song_texts, embeddings)
song_kb.save_local("faiss_indexes/song_kb")

# ========== 2. 构建 BM25 检索器并用 pickle 保存 ==========
# 设备库 BM25
device_bm25 = BM25Retriever.from_texts(device_texts)
with open("faiss_indexes/device_bm25.pkl", "wb") as f:
    pickle.dump(device_bm25, f)

# 销售库 BM25
sales_bm25 = BM25Retriever.from_texts(sales_texts)
with open("faiss_indexes/sales_bm25.pkl", "wb") as f:
    pickle.dump(sales_bm25, f)

# 歌曲库 BM25 - 使用与 FAISS 完全相同的格式
song_texts_for_bm25 = []
for item in song_data:
    combined = f"{item['title']} {item['artist']} {item['genre']} {item['description']}"
    song_texts_for_bm25.append(combined)
song_bm25 = BM25Retriever.from_texts(song_texts_for_bm25)
with open("faiss_indexes/song_bm25.pkl", "wb") as f:
    pickle.dump(song_bm25, f)

print("✅ 所有索引构建完成！")
print(f"   设备库: {len(device_texts)} 条")
print(f"   歌曲库: {len(song_texts)} 首")
print(f"   销售库: {len(sales_texts)} 条")