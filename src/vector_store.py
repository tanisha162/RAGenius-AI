from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


class HybridRetriever:

    def __init__(self, documents):

        self.documents = documents

        # FAISS semantic search
        embeddings = get_embeddings()

        self.vector_store = FAISS.from_documents(
            documents,
            embeddings
        )

        # BM25 keyword search
        tokenized_documents = [
            doc.page_content.lower().split()
            for doc in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)


    def search(self, query, k=5):

        # Semantic search using FAISS
        semantic_docs = self.vector_store.similarity_search(
            query,
            k=k
        )

        # Keyword search using BM25
        tokenized_query = query.lower().split()

        bm25_scores = self.bm25.get_scores(
            tokenized_query
        )

        # Get best BM25 document indexes
        top_indexes = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:k]

        keyword_docs = [
            self.documents[i]
            for i in top_indexes
        ]

        # Combine results and remove duplicates
        combined_docs = []
        seen_content = set()

        for doc in semantic_docs + keyword_docs:

            content_key = doc.page_content

            if content_key not in seen_content:

                combined_docs.append(doc)

                seen_content.add(content_key)

        # Return top results
        return combined_docs[:k]


def create_vector_store(documents):
    return HybridRetriever(documents)