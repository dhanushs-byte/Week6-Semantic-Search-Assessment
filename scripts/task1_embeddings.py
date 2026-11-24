from pathlib import Path
import sys, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.cluster import KMeans
from src.embeddings import LocalEmbedder, cosine_similarity, optional_api_embeddings, timed_encode

sentences = [
    "A customer cannot sign in after forgetting a password.",
    "The user needs help resetting their account password.",
    "Employees submit annual leave requests through the HR portal.",
    "Staff members can apply for vacation using the human resources system.",
    "The monthly invoice lists subscription charges and taxes.",
    "Customers can download their billing statement every month.",
    "A delivery package arrived later than the promised date.",
    "The courier tracking page shows the parcel location.",
    "The server experienced an outage during the maintenance window.",
    "The database backup completed successfully overnight."
]

print("\\nTASK 1: EMBEDDINGS AND COSINE SIMILARITY")
local = LocalEmbedder()
vectors, local_time = timed_encode(local, sentences)
print(f"Local model: all-MiniLM-L6-v2 | shape={vectors.shape} | time={local_time:.3f}s")
api_vectors, api_label = optional_api_embeddings(sentences)
if api_vectors is None:
    print(f"API comparison: {api_label}")
else:
    print(f"API model: {api_label} | shape={api_vectors.shape}")

tests = [([1,2,3],[2,4,6]), ([1,0],[0,1]), ([1,0],[-1,0])]
for a,b in tests:
    mine = cosine_similarity(a,b)
    theirs = float(sklearn_cosine([a],[b])[0][0])
    print(f"{a} vs {b} -> {mine:.1f} | sklearn: {theirs:.1f}")

matrix = vectors @ vectors.T
print("\\nSimilarity matrix (rounded):")
print(np.round(matrix, 3))
pairs = [(matrix[i,j], i, j) for i in range(len(sentences)) for j in range(i+1,len(sentences))]
print("\\nTop 3 similar pairs:")
for score,i,j in sorted(pairs, reverse=True)[:3]:
    print(f"{score:.4f}: [{i+1}] {sentences[i]} <> [{j+1}] {sentences[j]}")

print("\\nSemantic search (3 queries):")
queries = ["I cannot access my account", "I want to see what I paid this month", "Where is my delayed parcel?"]
for query in queries:
    q = local.encode([query])[0]
    scores = vectors @ q
    best = np.argsort(scores)[::-1][:3]
    print(f"Query: {query}")
    for idx in best:
        print(f"  {scores[idx]:.4f} | {sentences[idx]}")

print("\\nDuplicate detection (threshold=0.65):")
for score,i,j in sorted(pairs, reverse=True):
    if score >= 0.65:
        print(f"DUPLICATE-LIKE ({score:.3f}): '{sentences[i]}' / '{sentences[j]}'")

print("\\nK-Means clustering (3 named groups):")
labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(vectors)
names = {0:"Account and access", 1:"Billing and payments", 2:"Operations and delivery"}
# Name clusters by their dominant keyword theme rather than relying on arbitrary label numbers.
for label in sorted(set(labels)):
    members = [sentences[i] for i in range(len(sentences)) if labels[i] == label]
    joined = " ".join(members).lower()
    if any(w in joined for w in ["password","sign in","account"]): name = "Account and access"
    elif any(w in joined for w in ["invoice","billing","subscription","charges"]): name = "Billing and payments"
    else: name = "Operations, delivery and infrastructure"
    print(f"{name}:")
    for item in members: print("  -", item)

print("\\nChoosing local vs API (4 lines):")
print("1. Choose a local model when privacy, offline use and predictable cost matter.")
print("2. Choose an API when managed quality, easy scaling and less local compute matter.")
print("3. Compare latency, embedding quality, dimensions, rate limits and total cost.")
print("4. Keep the same embedding model/version for documents and queries in production.")
