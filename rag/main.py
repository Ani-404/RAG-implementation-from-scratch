# main file that takes the query and outputs the final answer

import argparse
from utils import get_documents
from rag_pipeline import RAGPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAGPipelineArguments')
    parser.add_argument('-q', '--query',
                        help    = "User's query",
                        type    = str,
                        default = None)
    parser.add_argument('--method',
                        help    = 'Retrieval method: semantic, keyword, faiss, hybrid, hybrid_rrf (default: hybrid_rrf)',
                        type    = str,
                        choices = ['semantic', 'keyword', 'faiss', 'hybrid', 'hybrid_rrf'],
                        default = 'hybrid_rrf')
    parser.add_argument('--rerank',
                        help    = 'Enable Cross-Encoder reranking on retrieved chunks',
                        action  = 'store_true')
    parser.add_argument('--top_k',
                        help    = 'Number of context chunks to retrieve (default: 3)',
                        type    = int,
                        default = 3)
    parser.add_argument('--embedder',
                        help    = 'Choose the embedding model. Default: "sentence-transformers/all-MiniLM-l6-v2"',
                        type    = str,
                        default = "sentence-transformers/all-MiniLM-l6-v2")
    parser.add_argument('--generator',  
                        help    = 'Choose the generator LLM. Default: "HuggingFaceTB/SmolLM2-360M-Instruct"',
                        type    = str,
                        default = "HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument('--doc_path',
                        help    = 'Path of stored text-based documents.',
                        type    = str,
                        default = None)
    args = parser.parse_args()

    # Setup documents (uses local documents directory by default)
    documents = get_documents(doc_path=args.doc_path) if args.doc_path else get_documents()

    # Setup RAG pipeline
    rag = RAGPipeline(embedding_model=args.embedder, generator_model=args.generator)
    
    # Create knowledge base
    rag.create_knowledge_base(documents, chunking_method="recursive", chunk_size=256, overlap=20)

    query = "What is the battery life of the AlphaTech SmartWatch X100, and does it support fast charging?" if args.query is None else args.query

    context = rag.similarity_search(query, method=args.method, top_k=args.top_k, rerank=args.rerank)

    print(f"\nRetrieved Context ({args.method.upper()}{' + Reranker' if args.rerank else ''}):\n", context)

    response = rag.generate_response(query, context)

    print("\nUser: ", query)
    print("Assistant: ", response)