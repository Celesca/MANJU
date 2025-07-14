@echo off
echo Installing RAG (Retrieval-Augmented Generation) dependencies...
echo.

echo Installing ChromaDB for vector database...
pip install chromadb>=0.4.0

echo Installing sentence-transformers for embeddings...
pip install sentence-transformers>=2.2.0

echo.
echo RAG dependencies installed successfully!
echo.
echo You can now enable RAG in the voice chatbot sidebar.
echo The system will use paraphrase-multilingual-MiniLM-L12-v2 for multilingual embeddings.
echo.
pause
