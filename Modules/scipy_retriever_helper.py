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
    - Preprocess text for optimal sparse retrieval
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
        
        # Timing tracking
        self.sparse_chunk_times = []
        self.dense_chunk_times = []
        self.model_loading_time = 0.0
        self.embedding_computation_time = 0.0
        
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

    def initialize_sparse(self, return_stats: bool = False) -> Optional[Dict]:
        """
        Initialize sparse (BM25) retrieval with per-chunk timing analysis.
        
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
        
        print("Analyzing sparse (BM25) incremental chunk timing...")
        
        start_time = time.time()
        self.sparse_chunk_times = []
        accumulated_docs = []
        
        for i, doc in enumerate(tqdm(self.documents, desc="Adding chunks to BM25")):
            chunk_start_time = time.time()
            
            # Add current document to accumulated list
            accumulated_docs.append(doc)
            
            # Recreate BM25 retriever with all accumulated documents
            # This is how BM25Retriever actually works - it needs to be rebuilt
            retriever = BM25Retriever.from_documents(accumulated_docs, k=5)
            
            chunk_time = (time.time() - chunk_start_time) * 1000  # Convert to ms
            self.sparse_chunk_times.append(chunk_time)
        
        self.sparse_retriever = retriever
        total_time = time.time() - start_time
        
        print(f"Sparse analysis complete! {len(self.sparse_chunk_times)} chunk times recorded.")
        
        if return_stats:
            return {
                'total_time': total_time,
                'chunk_times': self.sparse_chunk_times,
                'time_per_document': total_time / len(self.documents),
                'avg_chunk_time': np.mean(self.sparse_chunk_times),
                'min_chunk_time': np.min(self.sparse_chunk_times),
                'max_chunk_time': np.max(self.sparse_chunk_times)
            }

    def initialize_dense(self, return_stats: bool = False) -> Optional[Dict]:
        """
        Initialize dense (FAISS) retrieval with per-chunk timing analysis and GPU support.
        
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
        
        print("Analyzing dense (FAISS) incremental chunk timing...")
        if self.gpu_available:
            print("GPU acceleration available for FAISS")
        
        start_time = time.time()
        
        # Load model once and track time
        model_start_time = time.time()
        self.dense_model = HuggingFaceEmbeddings(model_name=self.dense_model_name)
        self.model_loading_time = time.time() - model_start_time
        
        self.dense_chunk_times = []
        vector_store = None
        embedding_start_time = time.time()
        
        for i, doc in enumerate(tqdm(self.documents, desc="Adding chunks to FAISS")):
            chunk_start_time = time.time()
            
            if vector_store is None:
                # First chunk - create new FAISS index
                vector_store = FAISS.from_documents([doc], self.dense_model)
                # Configure GPU if available
                if self.gpu_available:
                    try:
                        import faiss
                        # Move to GPU if possible
                        gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, vector_store.index)
                        vector_store.index = gpu_index
                    except Exception as e:
                        print(f"Warning: Could not move FAISS to GPU: {e}")
            else:
                # Add chunk to existing index
                vector_store.add_documents([doc])
            
            chunk_time = (time.time() - chunk_start_time) * 1000  # Convert to ms
            self.dense_chunk_times.append(chunk_time)
        
        self.embedding_computation_time = time.time() - embedding_start_time
        self.dense_retriever = vector_store
        total_time = time.time() - start_time
        
        print(f"Dense analysis complete! {len(self.dense_chunk_times)} chunk times recorded.")
        
        if return_stats:
            return {
                'total_time': total_time,
                'model_loading_time': self.model_loading_time,
                'embedding_computation_time': self.embedding_computation_time,
                'chunk_times': self.dense_chunk_times,
                'time_per_document': total_time / len(self.documents),
                'avg_chunk_time': np.mean(self.dense_chunk_times),
                'min_chunk_time': np.min(self.dense_chunk_times),
                'max_chunk_time': np.max(self.dense_chunk_times),
                'gpu_used': self.gpu_available
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
            raise ValueError("Sparse retriever not initialized. Call initialize_sparse() first.")
        
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
            raise ValueError("Dense retriever not initialized. Call initialize_dense() first.")
        
        start_time = time.time()
        retriever = self.dense_retriever.as_retriever(search_kwargs={"k": top_k})
        results = retriever.invoke(query)
        query_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return results, query_time
