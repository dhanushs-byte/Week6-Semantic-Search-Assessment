from pathlib import Path
import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embeddings import LocalEmbedder
from src.chunking import build_chunks
from src.vector_store import QdrantStore
from src.evaluation import retrieval_metrics

ROOT=Path(__file__).resolve().parents[1]
paths=sorted((ROOT/"data"/"documents").glob("*.txt"))
if len(paths)<5:
    raise SystemExit("Task 4 expects the five sample documents in data/documents.")
embedder=LocalEmbedder()
questions=[
    ("What is the purpose of the employee leave policy?", paths[0].name),
    ("How can a worker request annual leave?", paths[0].name),
    ("What is the approval process for leave?", paths[0].name),
    ("How should a password be reset?", paths[1].name),
    ("What are the account security rules?", paths[1].name),
    ("How do employees protect their accounts?", paths[1].name),
    ("How is a customer invoice generated?", paths[2].name),
    ("Where can a customer find billing details?", paths[2].name),
    ("What does the refund policy explain?", paths[3].name),
    ("When can a customer request a refund?", paths[3].name),
    ("How is a support ticket prioritized?", paths[4].name),
    ("What information should a support request include?", paths[4].name),
    ("Who handles urgent technical incidents?", paths[4].name),
    ("What are the customer service response targets?", paths[4].name),
    ("How can a user contact the support team?", paths[4].name),
]
def run_config(size):
    chunks=build_chunks(paths,strategy="recursive",size=size,overlap=min(50,size//4))
    store=QdrantStore(in_memory=True,collection=f"eval_{size}")
    store.upsert(chunks,embedder.encode([c["text"] for c in chunks]))
    rankings=[]
    for query,_ in questions:
        vector=embedder.encode([query])[0]
        rankings.append(store.search(vector,top_k=5))
    metrics=retrieval_metrics(rankings,[expected for _,expected in questions])
    store.close()
    return metrics

print("\\nTASK 4: RETRIEVAL EVALUATION")
a=run_config(256); b=run_config(512)
print("15 evaluation queries | expected source labels are defined in this script")
for name,metrics in [("Config A (chunk=256)",a),("Config B (chunk=512)",b)]:
    print(name, " | ".join(f"{k}={v:.3f}" for k,v in metrics.items()))
winner="A" if a["mrr"]>b["mrr"] else "B"
print(f"Decision: Config {winner} is better by MRR on this sample. Re-run on your own documents before making a production choice.")
print("\\nStart API with: uvicorn src.api:app --reload")
print("Swagger docs: http://127.0.0.1:8000/docs")
