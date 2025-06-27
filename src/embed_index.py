import pickle
import faiss
from sentence_transformers import SentenceTransformer

def load_chunks(chunks_file: str = "chunks.pkl"):
    """
    Load the list of chunk dicts created by ingest.py
    """
    with open(chunks_file, "rb") as f:
        return pickle.load(f)

def build_index(
    corpus: list[dict],
    model_name: str = "all-mpnet-base-v2",
    index_path: str = "faiss_index.index",
    corpus_path: str = "corpus.pkl"
):
    """
    1. Embed each chunk with SentenceTransformer.
    2. Normalize embeddings for cosine sim.
    3. Build a FAISS IndexFlatIP.
    4. Persist index and corpus metadata.
    """
    # load model
    model = SentenceTransformer(model_name)

    # texts → embeddings
    texts = [c["text"] for c in corpus]
    print(f"Encoding {len(texts)} chunks …")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # cosine similarity → inner product on L2-normalized vectors
    faiss.normalize_L2(embeddings)

    # build & populate index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors.")

    # save index and corpus
    faiss.write_index(index, index_path)
    with open(corpus_path, "wb") as f:
        pickle.dump(corpus, f)
    print(f"Index saved to {index_path}")
    print(f"Corpus metadata saved to {corpus_path}")

if __name__ == "__main__":
    corpus = load_chunks("chunks.pkl")
    build_index(corpus)
