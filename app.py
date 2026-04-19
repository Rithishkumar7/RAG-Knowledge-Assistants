from src.data_loader import load_all_documents
from src.vector_store import FaissVectorStore
from src.search import RAGSearch

# Example usage
if __name__ == "__main__":
    store = FaissVectorStore("faiss_store")
    if store.exists():
        store.load()
    else:
        print("[INFO] No saved Faiss index found. Building a new vector store...")
        docs = load_all_documents("data")
        store.build_from_documents(docs)
    rag_search = RAGSearch()
    query = input("Enter your query: ").strip() or "tell about rithish?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
