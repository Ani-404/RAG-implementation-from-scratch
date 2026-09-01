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
    parser.add_argument('-i', '--interactive',
                        help    = 'Start an interactive continuous Q&A session in the terminal',
                        action  = 'store_true')
    args = parser.parse_args()

    # Setup documents (uses local documents directory by default)
    documents = get_documents(doc_path=args.doc_path) if args.doc_path else get_documents()

    # Setup RAG pipeline
    rag = RAGPipeline(embedding_model=args.embedder, generator_model=args.generator)
    
    # Create knowledge base with 500 character chunks for rich semantic context
    rag.create_knowledge_base(documents, chunking_method="recursive", chunk_size=500, overlap=50)

    def answer_query(user_query: str):
        context = rag.similarity_search(user_query, method=args.method, top_k=args.top_k, rerank=args.rerank)
        response = rag.generate_response(user_query, context)
        print("\n" + "-" * 60)
        print(f"Assistant: {response}")
        print("-" * 60)
        print(f"[Retrieved {len(context)} chunks using {args.method.upper()}{' + RERANKER' if args.rerank else ''}]")

    if args.interactive:
        print("\n" + "=" * 60)
        print("  RAG Interactive Mode (Type 'exit' or 'quit' to stop)")
        print("=" * 60)
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit", "q"):
                    print("Exiting interactive mode.")
                    break
                answer_query(user_input)
            except (KeyboardInterrupt, EOFError):
                print("\nSession ended.")
                break
    else:
        query = "What is the battery life of the AlphaTech SmartWatch X100, and does it support fast charging?" if args.query is None else args.query
        print(f"\nUser: {query}")
        answer_query(query)