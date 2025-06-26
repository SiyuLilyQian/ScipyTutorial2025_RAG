# SciPy 2025 RAG Tutorial: Complete Hands-On Workshop

## Overview

This is the complete hands-on workshop for the **SciPy 2025 Conference RAG Tutorial**. You'll build production-ready Retrieval-Augmented Generation (RAG) systems from scratch using real scientific papers, learning everything from basic document processing to advanced reranking techniques.

## What You'll Master

This comprehensive tutorial covers the full RAG pipeline:

1. **LLM Setup & Integration**: Configure open-source language models
2. **Document Chunking**: Master 4 different chunking strategies for optimal retrieval
3. **Retrieval Systems**: Build both sparse (BM25) and dense (FAISS) retrievers
4. **RAG Integration**: Connect retrievers with LLMs for question-answering
5. **Advanced Techniques**: Implement hybrid retrievers and cross-encoder rerankers
6. **Interactive Apps**: Deploy your RAG system with Gradio interfaces

## 📁 Complete Project Structure

```
├── Modules/
│   ├── scipy_retriever_helper.py    # Basic retrieval class for both sparse and dense methods
│   └── scipy_chunking_helper.py     # 4 chunking methods (fixed, recursive, semantic, improved)
├── Demo_Notebooks/
│   ├── Scipy_2025_RAG_start_here.ipynb    # MAIN TUTORIAL - Start Here!
│   └── Chunking_Demo.ipynb                # Deep dive into chunking strategies
├── Data/
│   ├── selected_files_scipy/        # 90+ real scientific papers (arXiv format)
│   └── Scipy_RAG_QA.json            # 500+ curated Q&A pairs for testing
└── README.md                        # This comprehensive guide
```

## Quick Start Guide

### Prerequisites
- Google Colab (recommended) or Jupyter environment
- GPU access (optional but recommended for speed)

### Step 1: Begin Tutorial
**START HERE:** Open `Demo_Notebooks/Scipy_2025_RAG_start_here.ipynb`

This is your main tutorial notebook with:
- Complete setup instructions
- Step-by-step RAG implementation
- Interactive exercises with questions
- Gradio app deployment
- Advanced techniques (hybrid retrieval, reranking)

### Step 2: Explore Specialized Notebooks
- **Chunking Deep Dive**: `Chunking_Demo.ipynb` - Compare 4 chunking methods

## 📚 Tutorial Dataset: SPIQA Papers (Adapted Subset)

### What's Included
- **100 Scientific Papers**: Real arXiv papers from computer science and AI research
- **100 Q&A Pairs**: Professionally curated questions and answers about the papers
- **Diverse Topics**: Machine learning, computer vision, NLP, and more

### Dataset Citation
```bibtex
@article{pramanick2024spiqa,
    title={SPIQA: A Dataset for Multimodal Question Answering on Scientific Papers},
    author={Pramanick, Shraman and Chellappa, Rama and Venugopalan, Subhashini},
    journal={NeurIPS},
    year={2024}
}
```

**Source:** https://huggingface.co/datasets/google/spiqa

## SciPy 2025 Conference Tutorial

### Event Details
**"Retrieval Augmented Generation (RAG) for LLMs"**  
📅 **When:** July 7, 2025, 1:30–5:30 PM (US/Pacific)  
📍 **Where:** Ballroom C, SciPy 2025 Conference

### Why RAG Matters
Large Language Models face critical limitations:
- **Hallucinations**: Generating convincing but false information
- **Knowledge Cutoffs**: Limited to training data timeframes
- **Domain Expertise**: Poor performance on specialized topics

**RAG provides the solution** by grounding LLM responses in retrieved factual content.

### Tutorial Learning Outcomes
Participants will:
- Understand RAG architecture and core concepts
- Build complete RAG systems using open-source tools
- Master chunking strategies for different document types
- Implement and compare retrieval methods
- Deploy interactive RAG applications
- Apply advanced techniques (hybrid search, reranking)
- Evaluate and optimize RAG system performance

### Target Audience
- **Data Scientists** working with LLMs and knowledge systems
- **ML Engineers** building AI-powered search and QA applications
- **Researchers** needing reliable AI tools for domain-specific tasks
- **Software Engineers** developing intelligent applications

## 👥 Tutorial Development Team

Created and maintained by:
- **Sukhada**
- **Emily** 
- **Lily**
- **Antoni**

## Real-World Applications

This tutorial prepares you to build:
- **Scientific Literature Search**: AI-powered research assistants
- **Enterprise Knowledge Systems**: Company-specific QA bots
- **Educational Tools**: Intelligent tutoring systems with reliable sources
- **Technical Support**: Context-aware customer service bots
- **Legal Research**: Case law and regulation query systems
- **Medical Assistants**: Evidence-based clinical decision support

## Getting Started Now

1. **Clone/Download** this repository
2. **Open** `Demo_Notebooks/Scipy_2025_RAG_start_here.ipynb` in Colab
3. **Follow** the step-by-step instructions
4. **Experiment** with the 500+ included Q&A pairs
5. **Build** your own RAG applications!

## What You'll Build

By completing this tutorial, you'll have:
- A fully functional RAG system processing 90+ scientific papers
- Multiple chunking strategies with performance comparisons
- Sparse and dense retrieval implementations
- An interactive Gradio app for real-time Q&A
- Advanced hybrid retrieval with reranking
- Production-ready code for your own projects

**Ready to transform how AI accesses and uses knowledge? Let's build some RAG systems!**
