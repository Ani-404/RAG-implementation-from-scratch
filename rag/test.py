from utils import get_documents
from rag_pipeline import RAGPipeline

# Initialize the RAG pipeline
rag = RAGPipeline(embedding_model = 'sentence-transformers/all-MiniLM-l6-v2',
                  generator_model = 'HuggingFaceTB/SmolLM2-360M-Instruct')



with open(r"C:\Users\anime\OneDrive\Desktop\RAG-implementation-from-scratch\rag\documents\employee_handbook.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = rag.chunk_text(
    text=text,
    doc_name="NovaTech Employee Handbook",
    doc_id="doc_employee",
    method="recursive",
    chunk_size=500,
    overlap=50
)

for chunk in chunks:
    print(f"ID: {chunk['id']}")
    print(f"Text: {chunk['text'][:200]}")  # Print preview of first 200 characters
    print("-" * 80)
