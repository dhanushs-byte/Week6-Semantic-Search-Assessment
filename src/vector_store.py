"""Qdrant local vector store wrapper."""
from __future__ import annotations

import os
from pathlib import Path
from qdrant_client import QdrantClient, models


class QdrantStore:
    def __init__(self, path: str | None = None, collection: str = "week6_chunks",
                 dimension: int = 384, in_memory: bool = False):
        if in_memory:
            self.client = QdrantClient(":memory:")
        else:
            storage_path = path or os.getenv("QDRANT_PATH", "./storage/qdrant")
            Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=storage_path)
        self.collection = collection
        self.dimension = dimension
        if not self.client.collection_exists(collection_name=collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=dimension, distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100)
                )
            )

    def upsert(self, chunks: list[dict], vectors):
        points = []
        for chunk, vector in zip(chunks, vectors):
            payload = {**chunk["metadata"], "text": chunk["text"]}
            points.append(models.PointStruct(
                id=chunk["id"], vector=vector.tolist(), payload=payload
            ))
        if points:
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count

    def search(self, vector, top_k: int = 5, doc_type: str | None = None,
               min_score: float | None = None, min_page: int | None = None):
        conditions = []
        if doc_type:
            conditions.append(models.FieldCondition(
                key="doc_type", match=models.MatchValue(value=doc_type.lower().lstrip("."))
            ))
        if min_page is not None:
            conditions.append(models.FieldCondition(
                key="page", range=models.Range(gte=min_page)
            ))
        query_filter = models.Filter(must=conditions) if conditions else None
        hits = self.client.query_points(
            collection_name=self.collection, query=vector.tolist(),
            query_filter=query_filter, limit=top_k, with_payload=True
        ).points
        results = []
        for hit in hits:
            if min_score is not None and hit.score < min_score:
                continue
            payload = hit.payload or {}
            results.append({
                "id": str(hit.id), "source": payload.get("source"),
                "page": payload.get("page"), "section": payload.get("section"),
                "score": float(hit.score), "text": payload.get("text", ""),
                "doc_type": payload.get("doc_type")
            })
        return results

    def delete_source(self, source: str):
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[
                    models.FieldCondition(key="source", match=models.MatchValue(value=source))
                ])
            ), wait=True
        )

    def close(self):
        self.client.close()
