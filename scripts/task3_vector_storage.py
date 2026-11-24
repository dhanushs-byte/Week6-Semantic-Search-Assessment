from pathlib import Path
import sys, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import faiss
from src.embeddings import LocalEmbedder
from src.chunking import build_chunks
from src.vector_store import QdrantStore

print("\\nTASK 3: VECTOR STORAGE AND ANN")
print("ANN: Approximate Nearest Neighbor search finds close vectors without comparing every vector.")
print("It trades a small amount of recall for lower search latency on large collections.")
print("HNSW: Hierarchical Navigable Small World is a graph-based ANN index.")
print("It builds layered proximity graphs so queries can navigate toward nearby vectors quickly.")
print("M, efConstruction and efSearch tune graph connectivity, build cost and search quality.")

rng = np.random.default_rng(42)
data = rng.random((50000,384), dtype=np.float32)
faiss.normalize_L2(data)
query = rng.random((100,384), dtype=np.float32)
faiss.normalize_L2(query)

exact = faiss.IndexFlatIP(384)
exact.add(data)
t0=time.perf_counter(); exact_scores, exact_ids=exact.search(query,10); exact_ms=(time.perf_counter()-t0)*1000/len(query)

hnsw = faiss.IndexHNSWFlat(384,32,faiss.METRIC_INNER_PRODUCT)
hnsw.hnsw.efConstruction=100
hnsw.hnsw.efSearch=64
t0=time.perf_counter(); hnsw.add(data); build_s=(time.perf_counter()-t0)
t0=time.perf_counter(); approx_scores, approx_ids=hnsw.search(query,10); hnsw_ms=(time.perf_counter()-t0)*1000/len(query)
recall=np.mean([len(set(a)&set(e))/10 for a,e in zip(approx_ids,exact_ids)])
print(f"FAISS 50,000 x 384 | exact avg/query={exact_ms:.3f} ms | HNSW avg/query={hnsw_ms:.3f} ms")
print(f"HNSW build time={build_s:.2f}s | recall@10={recall:.3f}")

ROOT=Path(__file__).resolve().parents[1]
paths=sorted((ROOT/"data"/"documents").glob("*.txt"))
chunks=build_chunks(paths, strategy="recursive", size=512, overlap=50)
embedder=LocalEmbedder()
vectors=embedder.encode([c["text"] for c in chunks])
store=QdrantStore(path=str(ROOT/"storage"/"qdrant_task3"), collection="assessment_chunks")
store.upsert(chunks,vectors)
print(f"\\nQdrant points stored: {store.count()} | distance=Cosine | HNSW configured at collection creation")
q=embedder.encode(["What does the policy say about employee leave?"])[0]
print("Filtered search (doc_type=txt, page >= 1):")
for hit in store.search(q,top_k=5,doc_type="txt",min_page=1):
    print(f"  {hit['source']} page {hit['page']} score={hit['score']:.3f}")
before=store.count()
store.upsert(chunks,vectors)
print(f"Re-upsert: before={before}, after={store.count()} (count unchanged)")
if chunks:
    source=chunks[0]["metadata"]["source"]
    store.delete_source(source)
    print(f"Deleted source '{source}'. Remaining points: {store.count()}")
store.close()

print("\\npgvector vs Qdrant:")
print("| Topic | pgvector | Qdrant |")
print("| Setup | PostgreSQL + extension | Standalone server or local client |")
print("| Filtering | SQL WHERE clauses | Payload filters |")
print("| Scale | Fits existing Postgres workloads | Purpose-built vector search service |")
print("| Choose when | Relational data and vectors belong together | Dedicated vector search and payload filtering are priorities |")
