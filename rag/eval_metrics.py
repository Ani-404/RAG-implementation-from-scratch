from rag_pipeline import RAGPipeline
from utils import get_documents
import time

# test_queries = [
#     {
#         "query": "What behavior is expected of employees at NovaTech?",
#         "relevant_ids": ['1_1']
#     },
#     {
#         "query": "How should employees handle confidential company information?",
#         "relevant_ids": ['1_2']
#     },
#     {
#         "query": "Who is the target audience for GreenEarth’s solar panel campaign?",
#         "relevant_ids": ['2_1']
#     },
#     {
#         "query": "What are the key health monitoring features of the SmartWatch X100?",
#         "relevant_ids": ['3_1']
#     },
#     {
#         "query": "Which devices are compatible with the SmartWatch X100?",
#         "relevant_ids": ['3_2']
#     },
#     {
#         "query": "How long does the battery of the SmartWatch X100 last?",
#         "relevant_ids": ['3_3']
#     },
#     {
#         "query": "What anti-harassment policies are enforced at NovaTech?",
#         "relevant_ids": ['1_3']
#     },
#     {
#         "query": "How does GreenEarth promote their solar panels through partnerships?",
#         "relevant_ids": ['2_2']
#     },
#     {
#         "query": "What payment options are available with SmartWatch X100?",
#         "relevant_ids": ['3_3']
#     },
#     {
#         "query": "What is emphasized in GreenEarth’s campaign messaging?",
#         "relevant_ids": ['2_3']
#     }
# ]
if __name__ == "__main__":
    rag = RAGPipeline()  
    documents = get_documents()  # Your custom function that returns list of docs
    rag.create_knowledge_base(documents)


# def evaluate_ir_metrics(rag, test_queries, k=5):
#     precision_list = []
#     recall_list = []
#     mrr_list = []
#     topk_accuracy_list = []

#     for item in test_queries:
#         query = item["query"]
#         relevant_ids = set(item["relevant_ids"])

#         # Get top-k retrieved chunk IDs from your retriever
#         retrieved_chunks = rag.faiss_search(query, top_k=k)
#         retrieved_ids = [chunk["id"] for chunk in retrieved_chunks]  # ensure 'id' is part of your chunk metadata

#         retrieved_set = set(retrieved_ids)
#         intersection = retrieved_set & relevant_ids

#         # Precision@k
#         precision = len(intersection) / k
#         precision_list.append(precision)

#         # Recall@k
#         recall = len(intersection) / len(relevant_ids) if relevant_ids else 0
#         recall_list.append(recall)

#         # Top-k Accuracy
#         topk_accuracy = 1 if intersection else 0
#         topk_accuracy_list.append(topk_accuracy)

#         # MRR (Mean Reciprocal Rank)
#         rank = None
#         for idx, doc_id in enumerate(retrieved_ids):
#             if doc_id in relevant_ids:
#                 rank = idx + 1  # ranks start at 1
#                 break
#         mrr = 1 / rank if rank else 0
#         mrr_list.append(mrr)

#     return {
#         "Precision@k": sum(precision_list) / len(precision_list),
#         "Recall@k": sum(recall_list) / len(recall_list),
#         "Top-k Accuracy": sum(topk_accuracy_list) / len(topk_accuracy_list),
#         "MRR": sum(mrr_list) / len(mrr_list),
#         'query' : query,
#         'retrieved_ids': retrieved_ids,
#         'relevant_ids': relevant_ids
#     }



# results = evaluate_ir_metrics(rag, test_queries, k=5)
# print(results)

start = time.time()
retrieved_chunks = rag.faiss_search("How should employees handle confidential company information?", top_k=5)
elapsed = (time.time() - start) * 1000  # ms
print(f"FAISS search time: {elapsed:.2f} ms")