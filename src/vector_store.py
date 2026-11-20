from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


class HybridRetriever:

    def __init__(self, documents):

        if not documents:
            # A PDF with no extractable text (e.g. a scanned/image-only
            # PDF) yields zero chunks; FAISS and BM25 both crash on an
            # empty corpus (IndexError / ZeroDivisionError), so fail
            # here with a clear message instead.
            raise ValueError(
                "Cannot build a retriever from zero documents: "
                "no text could be extracted from the uploaded PDF(s)."
            )

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


    def invoke(self, query, config=None, **kwargs):
        """Retrieve documents for a query.

        app.py calls this method when the user asks a question, so it
        must exist (previously only ``search`` existed and every
        question crashed with AttributeError).
        """
        return self.search(query)

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

        # Combine results and remove duplicates, interleaving the
        # semantic and keyword lists so both legs of the hybrid
        # search can contribute. (Previously the semantic list was
        # placed first and the result truncated to k, so whenever
        # the corpus had >= k documents the keyword results were
        # always cut off and BM25 never affected the answer.)
        combined_docs = []
        seen_content = set()

        for semantic_doc, keyword_doc in zip(
            semantic_docs, keyword_docs
        ):
            for doc in (semantic_doc, keyword_doc):
                content_key = doc.page_content

                if content_key not in seen_content:
                    combined_docs.append(doc)
                    seen_content.add(content_key)

        # Either list may be longer than the other; append the tail.
        for doc in semantic_docs[len(keyword_docs):] + \
                keyword_docs[len(semantic_docs):]:
            content_key = doc.page_content

            if content_key not in seen_content:
                combined_docs.append(doc)
                seen_content.add(content_key)

        # Return top results
        return combined_docs[:k]


def create_vector_store(documents):
    return HybridRetriever(documents)