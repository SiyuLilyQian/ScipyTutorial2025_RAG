from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from typing import List, Dict, Optional, Tuple
import time
import numpy as np

class ScipyRetrieverHelper:
    """
    A helper class for scientific document retrieval using both sparse (BM25) and dense (FAISS) methods.
    
    This class provides functionality to:
    - Initialize both sparse and dense retrievers
    - Track per-chunk loading times for performance analysis
    - Support GPU acceleration where available
    """
    
    def __init__(self, dense_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the ScipyRetrieverHelper.
        
        Args:
            dense_model_name (str): Name of the HuggingFace model for dense embeddings
        """
        self.dense_model_name = dense_model_name
        self.dense_model = None
        self.dense_retriever = None
        self.sparse_retriever = None
        self.documents: List[Document] = []
        
        # Timing tracking - only chunk times
        self.sparse_chunk_times = []
        self.dense_chunk_times = []
        
        # GPU availability check
        self.gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self) -> bool:
        """
        Check if GPU is available for FAISS operations.
        
        Returns:
            bool: True if GPU is available, False otherwise
        """
        try:
            import faiss
            return faiss.get_num_gpus() > 0
        except ImportError:
            return False

    def _to_langchain_docs(self, chunks: List[Dict[str, str]]) -> List[Document]:
        """
        Convert chunks to LangChain documents without preprocessing.
        
        Args:
            chunks (List[Dict[str, str]]): List of chunk dictionaries
            
        Returns:
            List[Document]: Documents for both sparse and dense retrieval
        """
        docs = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["content"], 
                metadata=chunk.get("metadata", {})
            )
            docs.append(doc)
        return docs

    def add_documents(self, chunks: List[Dict[str, str]]) -> None:
        """
        Add documents to document collection for both sparse and dense retrieval.
        
        Args:
            chunks (List[Dict[str, str]]): List of chunk dictionaries containing content and metadata
        """
        docs = self._to_langchain_docs(chunks)
        self.documents.extend(docs)

    def initialize_sparse(self) -> None:
        """
        Initialize sparse (BM25) retrieval with all documents at once.
        """
        if not self.documents:
            raise ValueError("No documents to index. Call add_documents() first.")
        
        self.sparse_retriever = BM25Retriever.from_documents(self.documents, k=5)

    def initialize_dense(self) -> None:
        """
        Initialize dense (FAISS) retrieval with all documents at once.
        """
        if not self.documents:
            raise ValueError("No documents to index. Call add_documents() first.")
        
        # Load model
        self.dense_model = HuggingFaceEmbeddings(model_name=self.dense_model_name)
        
        # Create FAISS vector store with all documents at once
        self.dense_retriever = FAISS.from_documents(self.documents, self.dense_model)
        
        # Configure GPU if available
        if self.gpu_available:
            try:
                import faiss
                gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.dense_retriever.index)
                self.dense_retriever.index = gpu_index
            except Exception:
                pass  # Silently fail GPU setup

    def analyze_sparse_incremental_cost(self, return_stats: bool = False) -> Optional[Dict]:
        """
        Analyze incremental cost of adding documents to sparse (BM25) retrieval.
        Each timing represents the total time to process 1, 2, 3, ... documents.
        
        Args:
            return_stats (bool): If True, return timing statistics
            
        Returns:
            Optional[Dict]: Timing statistics if return_stats is True
        """
        if not self.documents:
            raise ValueError("No documents to index. Call add_documents() first.")
        
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, desc="", total=None):
                return iterable
        
        self.sparse_chunk_times = []
        
        for i in tqdm(range(1, len(self.documents) + 1), desc="BM25 Incremental Analysis", unit="docs"):
            chunk_start_time = time.time()
            
            # Rebuild BM25 retriever from scratch with documents 0 to i-1
            # This timing represents: time to process i documents total
            current_docs = self.documents[:i]
            retriever = BM25Retriever.from_documents(current_docs, k=5)
            
            chunk_time = (time.time() - chunk_start_time) * 1000  # Convert to ms
            self.sparse_chunk_times.append(chunk_time)
        
        # Keep the final retriever
        self.sparse_retriever = retriever
        
        if return_stats:
            return {
                'chunk_times': self.sparse_chunk_times
            }

    def analyze_dense_incremental_cost(self, return_stats: bool = False) -> Optional[Dict]:
        """
        Analyze incremental cost of adding documents to dense (FAISS) retrieval.
        Each timing represents the total time to process 1, 2, 3, ... documents.
        
        Args:
            return_stats (bool): If True, return timing statistics
            
        Returns:
            Optional[Dict]: Timing statistics if return_stats is True
        """
        if not self.documents:
            raise ValueError("No documents to index. Call add_documents() first.")
        
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, desc="", total=None):
                return iterable
        
        # Load model once
        self.dense_model = HuggingFaceEmbeddings(model_name=self.dense_model_name)
        
        self.dense_chunk_times = []
        
        for i in tqdm(range(1, len(self.documents) + 1), desc="FAISS Incremental Analysis", unit="docs"):
            chunk_start_time = time.time()
            
            # Rebuild FAISS vector store from scratch with documents 0 to i-1
            # This timing represents: time to process i documents total
            current_docs = self.documents[:i]
            vector_store = FAISS.from_documents(current_docs, self.dense_model)
            
            # Configure GPU if available
            if self.gpu_available:
                try:
                    import faiss
                    gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, vector_store.index)
                    vector_store.index = gpu_index
                except Exception:
                    pass  # Silently fail GPU setup
            
            chunk_time = (time.time() - chunk_start_time) * 1000  # Convert to ms
            self.dense_chunk_times.append(chunk_time)
        
        # Keep the final retriever
        self.dense_retriever = vector_store
        
        if return_stats:
            return {
                'chunk_times': self.dense_chunk_times
            }

    def retrieve_sparse(self, query: str, top_k: int = 5) -> Tuple[List[Document], float]:
        """
        Retrieve documents using sparse (BM25) method with timing.
        
        Args:
            query (str): Search query
            top_k (int): Number of top documents to retrieve
            
        Returns:
            Tuple[List[Document], float]: Retrieved documents and query time in milliseconds
        """
        if not self.sparse_retriever:
            raise ValueError("Sparse retriever not initialized. Call initialize_sparse() or analyze_sparse_incremental_cost() first.")
        
        start_time = time.time()
        self.sparse_retriever.k = top_k
        results = self.sparse_retriever.invoke(query)
        query_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return results, query_time

    def retrieve_dense(self, query: str, top_k: int = 5) -> Tuple[List[Document], float]:
        """
        Retrieve documents using dense (FAISS) method with timing.
        
        Args:
            query (str): Search query
            top_k (int): Number of top documents to retrieve
            
        Returns:
            Tuple[List[Document], float]: Retrieved documents and query time in milliseconds
        """
        if not self.dense_retriever:
            raise ValueError("Dense retriever not initialized. Call initialize_dense() or analyze_dense_incremental_cost() first.")
        
        start_time = time.time()
        retriever = self.dense_retriever.as_retriever(search_kwargs={"k": top_k})
        results = retriever.invoke(query)
        query_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return results, query_time