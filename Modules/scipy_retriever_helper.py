from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from typing import List, Dict

class ScipyRetrieverHelper:
    """
    A helper class for scientific document retrieval using both sparse (BM25) and dense (FAISS) methods.
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

    def add_documents(self, chunks: List[Dict[str, str]]) -> None:
        """
        Add documents to document collection for both sparse and dense retrieval.
        
        Args:
            chunks (List[Dict[str, str]]): List of chunk dictionaries containing content and metadata
        """
        docs = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["content"],
                metadata=chunk.get("metadata", {})
            )
            docs.append(doc)
        self.documents.extend(docs)

    def initialize_sparse(self) -> None:
        """
        Initialize sparse (BM25) retrieval.
        """
        if not self.documents:
            raise ValueError("No documents to index. Call add_documents() first.")
        
        self.sparse_retriever = BM25Retriever.from_documents(self.documents, k=5)

    def initialize_dense(self) -> None:
        """
        Initialize dense (FAISS) retrieval.
        """
        if not self.documents:
            raise ValueError("No documents to index. Call add_documents() first.")
        
        self.dense_model = HuggingFaceEmbeddings(model_name=self.dense_model_name)
        self.dense_retriever = FAISS.from_documents(self.documents, self.dense_model)

    def retrieve_sparse(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve documents using sparse (BM25) method.
        
        Args:
            query (str): Search query
            top_k (int): Number of top documents to retrieve
            
        Returns:
            List[Document]: Retrieved documents
        """
        if not self.sparse_retriever:
            raise ValueError("Sparse retriever not initialized. Call initialize_sparse() first.")
        
        self.sparse_retriever.k = top_k
        return self.sparse_retriever.invoke(query)

    def retrieve_dense(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve documents using dense (FAISS) method.
        
        Args:
            query (str): Search query
            top_k (int): Number of top documents to retrieve
            
        Returns:
            List[Document]: Retrieved documents
        """
        if not self.dense_retriever:
            raise ValueError("Dense retriever not initialized. Call initialize_dense() first.")
        
        retriever = self.dense_retriever.as_retriever(search_kwargs={"k": top_k})
        return retriever.invoke(query)
