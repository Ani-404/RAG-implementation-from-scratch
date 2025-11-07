
# RAG Implementation from Scratch

A complete **Retrieval-Augmented Generation (RAG)** pipeline implementation built from the ground up using custom retrievers, embeddings, and generators. This project demonstrates how to build a production-ready RAG system for document-based question answering without relying on external APIs.

## Overview

This repository provides a fully customizable RAG pipeline that handles document chunking, semantic and keyword-based retrieval, and response generation. It's designed for those who want to understand RAG mechanisms deeply or need a flexible system for their specific use cases.

### Key Features

- **Multiple Chunking Strategies**: Fixed, recursive, and document-based (Markdown) text splitting
- **Hybrid Retrieval**: Combine semantic search (embeddings), keyword-based search (BM25), and FAISS-powered vector search
- **Flexible Embedding Models**: Easily swap between different HuggingFace embedding models
- **Local Knowledge Base**: Save and load knowledge bases as JSON with embeddings stored as NumPy arrays
- **Extensible Generator**: Integrates with HuggingFace text generation models for response synthesis
- **Evaluation Metrics**: Built-in support for IR metrics (Precision, Recall, MRR, Top-k Accuracy)
- **Production-Ready**: Efficient indexing with FAISS and BM25 for fast retrieval

## Project Structure

```
RAG-implementation-from-scratch/
├── main.py                          # Entry point - query and response pipeline
├── rag_pipeline.py                  # Core RAGPipeline class
├── utils.py                         # Utility functions (document loading)
├── eval_metrics.py                  # Evaluation metrics and testing
├── test.py                          # Example usage and testing
├── documents/                       # Sample document directory
│   ├── employee_handbook.txt
│   ├── greenearth_marketing_strategy.txt
│   └── smartwatch_x100.txt
├── knowledge_base_*.json            # Saved knowledge base (auto-generated)
├── text_embeddings_*.npy            # Embeddings storage (auto-generated)
└── README.md                        # This file
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Dependencies

Install required packages:

```bash
pip install langchain-text-splitters langchain-community scikit-learn faiss-cpu rank-bm25 transformers torch numpy
```

For GPU support (FAISS), use `faiss-gpu` instead of `faiss-cpu`:

```bash
pip install faiss-gpu
```

## Quick Start

### 1. Basic Usage with Sample Documents

```bash
python main.py --query "What is the battery life of the SmartWatch X100?"
```

### 2. Customize Embeddings and Generator Models

```bash
python main.py \
  --query "What are NovaTech's anti-harassment policies?" \
  --embedder "sentence-transformers/all-MiniLM-l6-v2" \
  --generator "HuggingFaceTB/SmolLM2-360M-Instruct" \
  --doc_path "./documents"
```

### 3. Programmatic Usage

```python
from rag_pipeline import RAGPipeline
from utils import get_documents

# Initialize the RAG pipeline
rag = RAGPipeline(
    embedding_model='sentence-transformers/all-MiniLM-l6-v2',
    generator_model='HuggingFaceTB/SmolLM2-360M-Instruct'
)

# Load documents
documents = get_documents(doc_path='./documents')

# Create knowledge base
rag.create_knowledge_base(documents, chunking_method='recursive', chunk_size=256, overlap=20)

# Perform hybrid search
query = "What is the battery life of the SmartWatch X100?"
context = rag.similarity_search(query, method='hybrid', top_k=3)

# Generate response
response = rag.generate_response(query, context)
print(f"Query: {query}")
print(f"Response: {response}")
```

## Core Components

### RAGPipeline Class

The main class that orchestrates the entire RAG workflow.

#### Key Methods

**`__init__(embedding_model, generator_model)`**
- Initializes the pipeline with specified embedding and generation models
- Default: SmolLM2-360M-Instruct for both embedding and generation

**`chunk_text(text, doc_name, doc_id, method, chunk_size, overlap)`**
- Splits documents into retrievable chunks
- Supports: `fixed`, `recursive`, and `document` (Markdown) strategies
- Returns: List of chunk dictionaries with ID and text

**`create_knowledge_base(documents, chunking_method, chunk_size, overlap)`**
- Processes documents and builds searchable index
- Generates embeddings using the embedding model
- Creates FAISS and BM25 indices for fast retrieval
- Saves knowledge base to JSON and embeddings to NumPy

**`similarity_search(query, method, top_k)`**
- Retrieves relevant chunks using specified method
- Methods: `semantic` (embeddings), `keyword` (BM25), or `hybrid` (combination)
- Returns: Top-k most relevant text chunks

**`generate_response(query, retrieved_chunks, instructions)`**
- Generates answer using retrieved context
- Parameters allow customization of system instructions
- Returns: Generated response string

**`add_documents(documents, chunking_method, chunk_size, overlap)`**
- Incrementally adds new documents to existing knowledge base
- Updates embeddings and indices

**`save_knowledge_base(knowledge_base_path, embeddings_path)`**
- Persists knowledge base as JSON
- Saves embeddings as NumPy array

**`load_knowledge_base(knowledge_base_path, embeddings_path, faiss_index_path)`**
- Loads previously saved knowledge base and embeddings
- Rebuilds indices for retrieval

### Utils Module

**`get_documents(doc_path)`**
- Retrieves all .txt files from a directory
- Returns: List of document dictionaries with ID, name, and text content

## Configuration

### Chunking Strategies

1. **Fixed Chunking**: Splits text into fixed-size chunks
   - Best for: Uniform document types
   - Use: `method="fixed"`

2. **Recursive Chunking**: Splits recursively with semantic awareness
   - Best for: Mixed document types, better context preservation
   - Use: `method="recursive"` (recommended)

3. **Document-based**: Uses Markdown headers for splitting
   - Best for: Well-structured documents
   - Use: `method="document"`

### Retrieval Methods

- **Semantic Search**: Vector-based similarity using embeddings
- **Keyword Search**: BM25-based ranking for keyword relevance
- **Hybrid Search**: Combines semantic and keyword results with FAISS for optimal retrieval

### Supported Models

**Embedding Models** (HuggingFace):
- `sentence-transformers/all-MiniLM-l6-v2` (recommended for speed)
- `sentence-transformers/all-mpnet-base-v2` (better quality)
- `sentence-transformers/all-MiniLM-l6-v2` (default)

**Generator Models** (HuggingFace):
- `HuggingFaceTB/SmolLM2-360M-Instruct` (recommended for speed)
- `meta-llama/Llama-2-7b-hf`
- `mistralai/Mistral-7B-Instruct-v0.2`

## Example Workflow

```python
from rag_pipeline import RAGPipeline

# 1. Initialize
rag = RAGPipeline()

# 2. Create knowledge base from documents
documents = [
    {
        "id": "doc1",
        "name": "Company Handbook",
        "text": "Your document content here..."
    }
]
rag.create_knowledge_base(documents)

# 3. Query the system
context = rag.similarity_search("Your question here?", method="hybrid", top_k=3)

# 4. Generate response
response = rag.generate_response("Your question here?", context)
print(response)

# 5. Add more documents later
new_docs = [...]
rag.add_documents(new_docs)

# 6. Load saved knowledge base in future session
rag.load_knowledge_base(
    knowledge_base_path="knowledge_base_2025_06_18.json",
    embeddings_path="text_embeddings_2025_06_18.npy"
)
```

## Testing and Evaluation

Run the test suite to evaluate retrieval performance:

```bash
python test.py
```

Evaluate using IR metrics:

```bash
python eval_metrics.py
```

The evaluation module includes:
- **Precision@k**: Fraction of relevant documents in top-k results
- **Recall@k**: Fraction of all relevant documents retrieved
- **Mean Reciprocal Rank (MRR)**: Rank position of first relevant result
- **Top-k Accuracy**: Whether any relevant document appears in top-k

## Performance Considerations

### Speed Optimization

- **FAISS Indexing**: O(log n) search time vs O(n) for brute force
- **BM25 Ranking**: Fast keyword-based filtering for initial retrieval
- **Hybrid Search**: Combines strengths of both methods

### Memory Management

- Embeddings stored as NumPy arrays (compressed)
- FAISS uses efficient indexing (IndexFlatL2)
- Knowledge base stored as compact JSON

### Scaling

For large document collections:
1. Use smaller chunk sizes for finer granularity
2. Consider adding IVF (Inverted File) indexing in FAISS
3. Implement batch processing for knowledge base creation

## Troubleshooting

### Issue: "Module not found" errors

**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: Out of memory during embedding

**Solution**: Reduce chunk size or process documents in batches

### Issue: Low retrieval quality

**Solution**: 
- Try different embedding models
- Adjust chunk size and overlap parameters
- Switch to hybrid search method

### Issue: Slow FAISS search

**Solution**: Use faster FAISS indices (IndexIVFFlat for large datasets)

## Future Enhancements

- Integration with vector databases (Pinecone, Weaviate)
- Support for multimodal retrieval (images, PDFs)
- Advanced ranking algorithms (ColBERT, DPR)
- Fine-tuning capabilities for custom domains
- Web UI for interactive querying
- Evaluation dashboard with metrics visualization

## Contributing

Contributions are welcome! Areas for improvement:

- Adding support for additional document formats
- Implementing more advanced retrieval algorithms
- Performance optimizations
- Better error handling
- Documentation improvements

Please open an issue or submit a pull request with your suggestions.

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this implementation in your research or project, please cite:

```
@software{rag_implementation_2025,
  title={RAG Implementation from Scratch},
  author={Ani-404},
  year={2025},
  url={https://github.com/Ani-404/RAG-implementation-from-scratch}
}
```


