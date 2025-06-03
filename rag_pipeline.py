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



