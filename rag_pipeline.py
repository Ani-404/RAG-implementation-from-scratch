from typing import List, Dict, Optional
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from transformers import pipeline
from datetime import date
import numpy as np
import json, os


class RAGPipeline:

    """
    Core functionality
    Chunking: Splitting of the document into retrievable bits
    Embedding: Converting these text chunks into vector 
    Indexing: Storing embeddings for efficient retrieval.
    Retrieval: Fetching relevant chunks based on query similarity.
    Generation: Using retrieved context to generate responses.

        Key Features:
      * Supports multiple chunking strategies.
      * Offers keyword-based, semantic, and hybrid search.
      * Integrated with Hugging Face models for embedding and generation.
      * Supports local storage of knowledge base (JSON) and embeddings (NumPy).
      * Easily extendable to add new documents.

    """

    def __init__(self,
                 embedding_model = 'sentence-transformers/all-MiniLM-l6-v2',
                 generator_model = 'TinyLlama/TinyLlama-1.1B-Chat'):
        
        """
        Initialize the RAG pipeline.

        Args:
            embedding_model (str): Pretrained text embedding name.
            generator_model (str): Pretrained text generation model name.
        """

        # Setup embedding model
        self.embedding_model_name = embedding_model
        self.embedding_model      = None  # Initially set embedding_model to None; it will be loaded when creating/loading a KB.

        # Load text generation model
        self.generator = pipeline(
            "text-generation",
            model=generator_model,
            model_kwargs={"torch_dtype": "auto"},
            device_map="auto",
        )

        # Storage for indexed data
        self.index          = [] # Stores chunk metadata
        self.knowledge_base = {} # Stores document metadata
        self.embeddings     = np.array([])
        self.bm25           = None


    def chunk_text(self,
                   text: str,
                   doc_name: str,
                   doc_id: str,
                   method: str = "recursive",
                   chunk_size: int = 500,
                   overlap: int = 50,
                   markdown_headers: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """
        Splits text into smaller chunks and adds metadata to each chunk.

        Args:
            text (str): Document content.
            doc_name (str): Name of the document.
            doc_id (str): Unique document identifier.
            method (str): Chunking strategy: "fixed", "recursive", or "document".
            chunk_size (int): Size of chunks.
            overlap (int): Overlapping characters between chunks.
            markdown_headers (Optional[List[Dict[str, str]]]): Headers for Markdown/HTML chunking.

        Returns:
            List[Dict[str, str]]: List of chunks with metadata.
        """
        if method == 'fixed':
            splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        elif method == 'recursive':
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap= overlap)
        elif method == 'document':
            if markdown_headers:
                splitter = MarkdownHeaderTextSplitter(headers_to_split_on=markdown_headers)
            else:
                CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        else:
            raise ValueError("Unsupported chunking method. Choose from 'fixed', 'recursive', and 'document'.")

        chunks = splitter.split_text(text)
        return [{"id": f"{doc_id}_{i}", "text": chunk} for i, chunk in enumerate(chunks)]
    

    def create_knowledge_base(self,
                              documents: List[Dict[str, str]],
                              knowledge_base_path: str = f"knowledge_base_{date.today().strftime('%Y_%m_%d')}.json",
                              embeddings_path: str = f"text_embeddings_{date.today().strftime('%Y_%m_%d')}.npy",
                              chunking_method: str = "recursive",
                              chunk_size: int = 500,
                              overlap: int = 50,
                              markdown_headers: Optional[List[Dict[str, str]]] = None):
        """
        Creates a new knowledge base by processing and indexing documents.

        Args:
            documents (List[Dict[str, str]]): List of documents (each with "id", "name", and "text").
            knowledge_base_path (str): Path to save the knowledge base as JSON.
            embeddings_path (str): Path to save the embeddings as NumPy array.
            chunking_method (str): Chunking strategy.
            chunk_size (int): Size of chunks.
            overlap (int): Overlapping characters between chunks.
        """
        self.knowledge_base = {
            "embedding_model_name": self.embedding_model_name,
            "documents": []
        }
        self.index  = []
        chunk_texts = []
        
        for doc in documents:
            doc_id   = doc["id"]
            doc_name = doc["name"]
            chunks   = self.chunk_text(doc["text"], doc_name=doc_name, doc_id=doc_id, method=chunking_method,
                                       chunk_size=chunk_size, overlap=overlap, markdown_headers=markdown_headers)
            self.knowledge_base["documents"].append({
                "id": doc_id,
                "name": doc_name,
                "chunking_technique": chunking_method,
                "chunks": chunks
            })
            self.index.extend(chunks)
            chunk_texts.extend([chunk["text"] for chunk in chunks])
        
        # Load embedding model
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        # Compute text embeddings
        self.embeddings = np.array(self.embedding_model.embed_documents(chunk_texts))

        # Store BM25 index for hybrid retrieval
        tokenized_chunks = [chunk.lower().split() for chunk in chunk_texts]
        self.bm25 = BM25Okapi(tokenized_chunks)

        

