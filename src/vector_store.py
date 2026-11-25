import re

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi


def _tokenize(text):
    """Split text into lowercase word tokens, stripping punctuation.

    Plain ``str.split()`` keeps sentence punctuation attached to words,
    so "zebraxylophone." never matched the query word "zebraxylophone"
    (and a query ending in "?" never matched the bare word in a
    document) — the BM25 leg silently missed exact keyword hits
    whenever the keyword ended a sentence or carried punctuation.
    """
    return re.findall(r"\w+", text.lower())


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
            _tokenize(doc.page_content)
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
        tokenized_query = _tokenize(query)

        bm25_scores = self.bm25.get_scores(
            tokenized_query
        )

        # Get best BM25 document indexes — but only documents that
        # actually matched a query keyword (score > 0). When no query
        # token appears in any document BM25 scores every document
        # 0, and taking the top-k anyway returned the first k
        # documents in corpus order: arbitrary, irrelevant documents
        # that were interleaved into the results and displaced real
        # semantic hits.
        top_indexes = [
            i for i in sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True
            )
            if bm25_scores[i] > 0
        ][:k]

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
        #
        # A "duplicate" is the same chunk returned by both legs, so
        # dedupe on (content, source, page) — not on content alone.
        # The old content-only key silently dropped a genuinely
        # different chunk whenever two chunks had identical text
        # (e.g. a clause repeated on another page), so that page
        # never appeared in the results or the sources list.
        combined_docs = []
        seen_chunks = set()

        def _chunk_key(doc):
            metadata = doc.metadata or {}
            return (
                doc.page_content,
                metadata.get("source"),
                metadata.get("page"),
            )

        for semantic_doc, keyword_doc in zip(
            semantic_docs, keyword_docs
        ):
            for doc in (semantic_doc, keyword_doc):
                key = _chunk_key(doc)

                if key not in seen_chunks:
                    combined_docs.append(doc)
                    seen_chunks.add(key)

        # Either list may be longer than the other; append the tail.
        for doc in semantic_docs[len(keyword_docs):] + \
                keyword_docs[len(semantic_docs):]:
            key = _chunk_key(doc)

            if key not in seen_chunks:
                combined_docs.append(doc)
                seen_chunks.add(key)

        # Return top results
        return combined_docs[:k]


def create_vector_store(documents):
    return HybridRetriever(documents)