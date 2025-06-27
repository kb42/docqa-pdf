import os
import pickle
import fitz                     # PyMuPDF
from transformers import AutoTokenizer

def extract_text(pdf_path: str) -> str:
    """
    Load a PDF with PyMuPDF and return its full text.
    """
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    return "\n".join(pages)

def chunk_text(
    text: str,
    tokenizer,
    chunk_size: int = 512,
    overlap: int = 256
) -> list[str]:
    """
    Tokenize `text`, split into chunks of length `chunk_size` with `overlap`.
    Returns a list of text chunks.
    """
    # encode → list of token IDs (no special tokens)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(token_ids):
        end = start + chunk_size
        chunk_ids = token_ids[start:end]
        chunk = tokenizer.decode(chunk_ids)
        chunks.append(chunk)
        if end >= len(token_ids):
            break
        start += chunk_size - overlap
    return chunks

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract & chunk all PDFs in a directory"
    )
    parser.add_argument(
        "--input_dir", type=str, default="data/docs",
        help="Folder containing .pdf files"
    )
    parser.add_argument(
        "--model_name", type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="HuggingFace tokenizer to use"
    )
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap",   type=int, default=256)
    parser.add_argument(
        "--output_file", type=str, default="chunks.pkl",
        help="Where to save the list of chunk dicts"
    )
    args = parser.parse_args()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    corpus = []
    for fname in os.listdir(args.input_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(args.input_dir, fname)
        print(f"Ingesting {fname} ...")
        raw = extract_text(path)
        for i, chunk in enumerate(chunk_text(
            raw, tokenizer, args.chunk_size, args.overlap
        )):
            corpus.append({
                "doc_id": fname,
                "chunk_id": i,
                "text": chunk
            })

    # save to disk
    with open(args.output_file, "wb") as f:
        pickle.dump(corpus, f)
    print(f"Saved {len(corpus)} chunks → {args.output_file}")

if __name__ == "__main__":
    main()
