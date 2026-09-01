import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rag_pipeline import RAGPipeline
from utils import get_documents

# Ground-truth test queries with relevant chunk IDs ({doc_id}_{chunk_index}, 0-indexed)
test_queries = [
    {
        "query": "What behavior is expected of employees at NovaTech?",
        "relevant_ids": ["1_0", "1_1"]
    },
    {
        "query": "How should employees handle confidential company information?",
        "relevant_ids": ["1_2"]
    },
    {
        "query": "Who is the target audience for GreenEarth's solar panel campaign?",
        "relevant_ids": ["2_0", "2_1"]
    },
    {
        "query": "What are the key health monitoring features of the SmartWatch X100?",
        "relevant_ids": ["3_0", "3_1"]
    },
    {
        "query": "Which devices are compatible with the SmartWatch X100?",
        "relevant_ids": ["3_1", "3_2"]
    },
    {
        "query": "How long does the battery of the SmartWatch X100 last?",
        "relevant_ids": ["3_2", "3_3"]
    },
    {
        "query": "What anti-harassment policies are enforced at NovaTech?",
        "relevant_ids": ["1_3"]
    },
    {
        "query": "How does GreenEarth promote their solar panels through partnerships?",
        "relevant_ids": ["2_1", "2_2"]
    }
]


def retrieve_ids(rag: RAGPipeline, query: str, method: str = "faiss", top_k: int = 3, rerank: bool = False):
    """
    Retrieves top_k chunk IDs for a given query based on the selected method, with optional cross-encoder reranking.
    """
    fetch_k = top_k * 2 if rerank else top_k

    if method == "faiss":
        results = rag.faiss_search(query, top_k=fetch_k)
        ids = [c["id"] if isinstance(c, dict) else c for c in results]

    elif method == "semantic":
        query_embedding = np.array(rag.embedding_model.embed_query(query)).reshape(1, -1)
        sims = cosine_similarity(query_embedding, rag.embeddings)[0]
        top_indices = np.argsort(sims)[-fetch_k:][::-1]
        ids = [rag.index[i]["id"] for i in top_indices]

    elif method == "keyword":
        tokenized_query = query.lower().split()
        scores = rag.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-fetch_k:][::-1]
        ids = [rag.index[i]["id"] for i in top_indices]

    elif method == "hybrid":
        faiss_ids = retrieve_ids(rag, query, method="faiss", top_k=fetch_k)
        sem_ids = retrieve_ids(rag, query, method="semantic", top_k=fetch_k)
        kw_ids = retrieve_ids(rag, query, method="keyword", top_k=fetch_k)

        combined = []
        for doc_id in (faiss_ids + sem_ids + kw_ids):
            if doc_id not in combined:
                combined.append(doc_id)
        ids = combined[:fetch_k]

    elif method == "hybrid_rrf":
        faiss_ids = retrieve_ids(rag, query, method="faiss", top_k=fetch_k)
        sem_ids = retrieve_ids(rag, query, method="semantic", top_k=fetch_k)
        kw_ids = retrieve_ids(rag, query, method="keyword", top_k=fetch_k)

        rrf_scores = {}
        for ranked_list in [sem_ids, kw_ids, faiss_ids]:
            for rank, item in enumerate(ranked_list, start=1):
                rrf_scores[item] = rrf_scores.get(item, 0.0) + (1.0 / (60 + rank))
        ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:fetch_k]

    else:
        raise ValueError(f"Unknown retrieval method: {method}")

    # Apply Cross-Encoder reranking if enabled
    if rerank and ids:
        id_to_chunk = {c["id"]: c["text"] for c in rag.index}
        candidate_texts = [id_to_chunk[doc_id] for doc_id in ids if doc_id in id_to_chunk]
        if rag.reranker is None:
            from sentence_transformers import CrossEncoder
            rag.reranker = CrossEncoder(rag.reranker_model_name)
        scores = rag.reranker.predict([[query, t] for t in candidate_texts])
        sorted_indices = np.argsort(scores)[::-1]
        ids = [ids[i] for i in sorted_indices][:top_k]

    return ids[:top_k]


def evaluate_retrieval(rag: RAGPipeline, queries: list, method: str = "faiss", k: int = 3, rerank: bool = False):
    """
    Evaluates Precision@k, Recall@k, Hit Rate (Top-k Accuracy), MRR, and Latency.
    """
    precision_list = []
    recall_list = []
    hit_list = []
    mrr_list = []
    latencies = []

    for item in queries:
        query = item["query"]
        relevant_ids = set(item["relevant_ids"])

        # Measure retrieval latency
        t0 = time.perf_counter()
        retrieved_ids = retrieve_ids(rag, query, method=method, top_k=k, rerank=rerank)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        retrieved_set = set(retrieved_ids)
        intersection = retrieved_set & relevant_ids

        # Precision@k
        precision = len(intersection) / k if k > 0 else 0
        precision_list.append(precision)

        # Recall@k
        recall = len(intersection) / len(relevant_ids) if relevant_ids else 0
        recall_list.append(recall)

        # Hit Rate (Top-k Accuracy)
        hit = 1.0 if len(intersection) > 0 else 0.0
        hit_list.append(hit)

        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                mrr = 1.0 / rank
                break
        mrr_list.append(mrr)

    name = f"{method.upper()} + Rerank" if rerank else method.upper()
    return {
        "Method": name,
        f"Precision@{k}": sum(precision_list) / len(precision_list),
        f"Recall@{k}": sum(recall_list) / len(recall_list),
        "Hit Rate": sum(hit_list) / len(hit_list),
        "MRR": sum(mrr_list) / len(mrr_list),
        "Latency (ms)": sum(latencies) / len(latencies),
    }


if __name__ == "__main__":
    print("=" * 80)
    print("  Initializing RAG Pipeline and Indexing Knowledge Base...")
    print("=" * 80)

    rag = RAGPipeline()
    documents = get_documents()
    rag.create_knowledge_base(documents)

    k_val = 3
    configs = [
        ("keyword", False),
        ("semantic", False),
        ("faiss", False),
        ("hybrid", False),
        ("hybrid_rrf", False),
        ("hybrid_rrf", True),
    ]
    results = []

    print(f"\nEvaluating {len(test_queries)} queries across retrieval configurations (k={k_val})...\n")

    for method, rerank in configs:
        res = evaluate_retrieval(rag, test_queries, method=method, k=k_val, rerank=rerank)
        results.append(res)

    # Print summary table
    header = f"{'Method':<20} | {f'Precision@{k_val}':<14} | {f'Recall@{k_val}':<12} | {'Hit Rate':<10} | {'MRR':<8} | {'Latency (ms)':<12}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['Method']:<20} | "
            f"{r[f'Precision@{k_val}']:.4f}{'':<8} | "
            f"{r[f'Recall@{k_val}']:.4f}{'':<6} | "
            f"{r['Hit Rate']:.4f}{'':<4} | "
            f"{r['MRR']:.4f}{'':<2} | "
            f"{r['Latency (ms)']:<12.2f}"
        )
    print("-" * len(header))