from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter
)
import os
import nltk
import pypdf
from nltk import sent_tokenize
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from sentence_transformers import SentenceTransformer, util


def fixed_length_chunking(doc: Document, chunk_size: int = 200, chunk_overlap: int = 0, separator: str = ""):
    """
    Splits a single LangChain Document into smaller chunks using CharacterTextSplitter.
    Args:
        doc (Document): The input Document to split.
        chunk_size (int): Maximum size of each chunk in characters.
        chunk_overlap (int): Number of characters to overlap between chunks.
        separator (str): Separator used to split the text.
    Returns:
        List[Document]: A list of chunked Document objects with preserved metadata.
    """
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents([doc])


def recursive_chunking(
    doc: Document,
    chunk_size: int = 200,
    chunk_overlap: int = 0,
    separators = ["\n\n", "\n", ".", "!", "?", " "],
):
    """
    Chunk a LangChain Document using RecursiveCharacterTextSplitter.
    Args:
        doc (Document): The input Document to split.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap between chunks.
        separators (List[str]): Preferred split points.
    Returns:
        List[Document]: A list of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    chunks = splitter.split_documents([doc])
    return chunks


def semantic_chunking(    
    doc: Document,
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: int = 70
):
    """
    Chunk a LangChain Document using SemanticChunker.
    Args:
        doc (Document): The input Document to split.
        embedding_model_name (str): The embedding model to use.
        breakpoint_threshold_type (str): Type of threshold ("percentile" or "value").
        breakpoint_threshold_amount (int): Threshold amount.
    Returns:
        List[Document]: A list of semantically chunked Document objects.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    semantic_chunker = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount
    )
    chunks = semantic_chunker.split_documents([doc])
    return chunks


def semantic_chunking_improved(
    doc: Document,
    embedder: str = "all-mpnet-base-v2",
    max_chunk_size: int = 300,
    min_chunk_size: int = 100,
    sim_threshold: float = 0.7
):
    """
    Semantically chunk a single LangChain Document.
    Args:
        doc (Document): The input document to chunk.
        embedder (str): The embedding model to use.
        max_chunk_size (int): Max characters allowed per chunk.
        min_chunk_size (int): Merge to previous if chunk is below this.
        sim_threshold (float): Cosine similarity threshold to trigger a split.
    Returns:
        List[Document]: Chunked Document objects with inherited metadata.
    """
    # Get text and split into sentences
    text = doc.page_content
    sentences = sent_tokenize(text)

    # Compute sentence embeddings
    embedder = SentenceTransformer("all-mpnet-base-v2")
    sentence_embeddings = embedder.encode(sentences)

    chunks = []
    current_chunk = sentences[0]
    current_length = len(sentences[0])

    # Iterate through each sentence, starting from the second
    for i in range(1, len(sentences)):
        # Compare similarity with the previous sentence
        similarity = float(util.cos_sim(sentence_embeddings[i], sentence_embeddings[i - 1]))

        # Start a new chunk if similarity is low or adding the sentence would exceed max size
        if similarity < sim_threshold or current_length + len(sentences[i]) > max_chunk_size:
            # If the chunk is too small, merge it with the previous chunk
            if len(current_chunk) < min_chunk_size and chunks:
                chunks[-1].page_content += " " + current_chunk
            else:
                # Otherwise, store the current chunk as a Document
                chunks.append(Document(page_content=current_chunk, metadata=doc.metadata))

            # Reset chunk state
            current_chunk = sentences[i]
            current_length = len(sentences[i])
        else:
            # If similar and under size, add sentence to current chunk
            current_chunk += " " + sentences[i]
            current_length += len(sentences[i])

    # Handle final chunk
    if len(current_chunk) < min_chunk_size and chunks:
        chunks[-1].page_content += " " + current_chunk
    else:
        chunks.append(Document(page_content=current_chunk, metadata=doc.metadata))
    return chunks


def print_chunks(chunks, show_content=True):
    """
    Print a list of chunks (Document or str).
    Args:
        chunks (List[str] or List[Document]): The chunks to print.
        show_content (bool): Whether to print chunk content.
    """
    for i, chunk in enumerate(chunks, 1):
        content = chunk.page_content if hasattr(chunk, "page_content") else chunk
        print(f"Chunk {i} (length: {len(content)}):")
        if show_content:
            print(content)
        print("-" * 40 )