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
    

    



