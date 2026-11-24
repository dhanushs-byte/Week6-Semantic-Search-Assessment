from pathlib import Path
import sys, shutil, tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import build_chunks, extract_document, fixed_token_chunks, sentence_chunks, recursive_chunks, stable_chunk_id
from src.embeddings import LocalEmbedder
from src.vector_store import QdrantStore

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "documents"
paths = sorted(DOCS.glob("*.txt")) + sorted(DOCS.glob("*.pdf")) + sorted(DOCS.glob("*.docx"))
print("\\nTASK 2: CHUNKING, METADATA AND STABLE IDS")
print("Documents found:", len(paths))
if len(paths) < 5:
    raise SystemExit("Add at least five PDF, DOCX or TXT files under data/documents.")

sample = extract_document(paths[0])[0]["text"]
print("Chunk strategy comparison (first document/page):")
for name, pieces in [
    ("Fixed 512/50", fixed_token_chunks(sample,512,50)),
    ("Sentence", sentence_chunks(sample,100)),
    ("Recursive 512/50", recursive_chunks(sample,512,50))
]:
    print(f"  {name}: {len(pieces)} chunks")
sentences = sentence_chunks(sample, 12)
fixed = fixed_token_chunks(sample, 12, 0)
print("Sentence-safe first chunk:", sentences[0][:180] if sentences else "(none)")
print("Fixed-size first chunk (may cut a sentence):", fixed[0][:180] if fixed else "(none)")

print("\\nChunk-size experiment (10 sample questions; heuristic page coverage):")
for size in [128,256,512,1024]:
    chunks = build_chunks(paths, strategy="recursive", size=size, overlap=min(50,size//4))
    # A known-page question is considered retrievable if its source/page has at least one chunk.
    available = {(c["metadata"]["source"],c["metadata"]["page"]) for c in chunks}
    questions = [(p.name, 1) for p in paths[:5]] + [(p.name, 1) for p in paths[:5]]
    hits = sum(1 for q in questions if q in available)
    print(f"  size={size:4d} | chunks={len(chunks):3d} | hit-rate={hits/len(questions):.2f}")

chunks = build_chunks(paths, strategy="recursive", size=512, overlap=50)
print(f"\\nRecursive 512/50 total chunks: {len(chunks)}")
print("Metadata example:", chunks[0]["metadata"])
print("Citation:", __import__("src.chunking",fromlist=["describe_chunk"]).describe_chunk(chunks[0]["metadata"]))

# Demonstrate stable IDs and content update in a persistent Qdrant local store.
embedder = LocalEmbedder()
store_path = ROOT / "storage" / "task2_qdrant"
store = QdrantStore(path=str(store_path), collection="task2_chunks")
vectors = embedder.encode([c["text"] for c in chunks])
store.upsert(chunks, vectors)
count1 = store.count()
store.upsert(chunks, vectors)
count2 = store.count()
print(f"Index run 1: {count1} points | run 2: {count2} points (no growth)")

# Simulate one paragraph edit: same location-based ID is upserted, replacing its content/vector.
edited = dict(chunks[0])
edited["text"] = edited["text"] + " This sentence was added during the edit demonstration."
edited_vector = embedder.encode([edited["text"]])
store.upsert([edited], edited_vector)
count3 = store.count()
print(f"Run 3 (one chunk edited): {count3} points | replaced in place: {edited['id'] == chunks[0]['id']}")
store.close()
