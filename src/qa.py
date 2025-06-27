import os
import pickle
import argparse
import torch

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

import groq
from groq import Groq


def load_resources(
    index_path: str = "faiss_index.index",
    corpus_path: str = "corpus.pkl",
    embedder_name: str = "all-mpnet-base-v2"
):
    # load the faiss index for fast similarity search
    index = faiss.read_index(index_path)

    # load the preprocessed document chunks
    with open(corpus_path, "rb") as f:
        corpus = pickle.load(f)

    # load the sentence embedding model
    embedder = SentenceTransformer(embedder_name)

    return index, corpus, embedder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def retrieve_and_rerank(
    query: str,
    index,
    corpus: list[dict],
    embedder,
    faiss_k: int = 20,
    top_k: int = 5
):
    # embed the query using our sentence transformer
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    
    # find initial candidates using faiss
    scores, ids = index.search(q_emb, faiss_k)
    candidates = [corpus[i] for i in ids[0]]

    # rescore using cross-encoder for better relevance
    pairs = [[query, c["text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)
    for c, s in zip(candidates, rerank_scores):
        c["rerank_score"] = float(s)

    # sort by the new reranking scores
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # return only the top results
    return candidates[:top_k]


class ExtractiveQA:
    def __init__(self, model_name="deepset/roberta-base-squad2", device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
       
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # dyanmic thresh calc later
        self.no_answer_threshold = 0.0
        
    def _preprocess_inputs(self, question, context, max_length=512):
        # tokenize question and context together
        encoding = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        return encoding
    
    def _postprocess_answer(self, encoding, start_logits, end_logits, context):
        start_logits = start_logits.cpu().numpy()
        end_logits = end_logits.cpu().numpy()
        
        # get character positions for tokens
        offset_mapping = encoding["offset_mapping"][0].cpu().numpy()
        
        # identify where context starts in token sequence
        sequence_ids = encoding.sequence_ids(0)
        context_start = None
        context_end = None
        
        for i, seq_id in enumerate(sequence_ids):
            if seq_id == 1:
                if context_start is None:
                    context_start = i
                context_end = i
        
        # safety check when no context
        if context_start is None or context_end is None:
            return "", 0.0, False
        
        # convert logits to probabilities
        start_probs = torch.nn.functional.softmax(torch.tensor(start_logits), dim=-1)
        end_probs = torch.nn.functional.softmax(torch.tensor(end_logits), dim=-1)
        
        # calculate no-answer score using cls token
        null_start_logit = start_logits[0]
        null_end_logit = end_logits[0]
        null_score = null_start_logit + null_end_logit
        
        # search for best answer span within context
        best_non_null_score = -float('inf')
        best_start_idx = context_start
        best_end_idx = context_start
        
        # only consider reasonable answer lengths
        for start_idx in range(context_start, context_end + 1):
            for end_idx in range(start_idx, min(start_idx + 30, context_end + 1)):
                score = start_logits[start_idx] + end_logits[end_idx]
                if score > best_non_null_score:
                    best_non_null_score = score
                    best_start_idx = start_idx
                    best_end_idx = end_idx
        
        # compare best answer score vs no-answer score
        score_diff = best_non_null_score - null_score
        
        # calculate answer confidence
        answer_confidence = (start_probs[best_start_idx] * end_probs[best_end_idx]).item()
            
        # extract actual text using character offsets
        start_char = offset_mapping[best_start_idx][0]
        end_char = offset_mapping[best_end_idx][1]
        
        # handle empty answers
        if start_char == 0 and end_char == 0:
            return "", 0.0, False
            
        answer_text = context[start_char:end_char].strip()
        
        # skip whitespace-only answers
        if not answer_text or answer_text.isspace():
            return "", 0.0, False
            
        # adjust thresholds based on answer length
        min_confidence_threshold = 0.1
        if len(answer_text.split()) == 1:
            min_confidence_threshold = 0.15
        elif len(answer_text) < 5:
            min_confidence_threshold = 0.2
            
        # final decision on answerability
        is_answerable = (score_diff > 0.0 and answer_confidence > min_confidence_threshold)
        
        if is_answerable:
            return answer_text, answer_confidence, True
        
        # if not answerable, return no-answer confidence
        no_answer_confidence = torch.nn.functional.softmax(torch.tensor([null_score, best_non_null_score]), dim=-1)[0].item()
        return "", no_answer_confidence, False
    
    def extractive_qa(self, question, context, max_length=512, return_confidence=False, confidence_threshold=None):
        # handle empty inputs
        if not question or not context:
            if return_confidence:
                return "", 0.0, False
            return ""
        
        # clean inputs
        question = question.strip()
        context = context.strip()
        
        # tokenize and prepare model inputs
        encoding = self._preprocess_inputs(question, context, max_length) # causing inaccuraies - look into it
        
        # move to gpu? 
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # run model inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]
        
        # process model outputs
        answer_text, confidence, is_answerable = self._postprocess_answer(
            encoding, start_logits, end_logits, context
        )
        
        # apply custom threshold if we have it
        if confidence_threshold is not None:
            is_answerable = confidence >= confidence_threshold and answer_text.strip() != ""
            if not is_answerable:
                answer_text = ""
        
        if return_confidence:
            return answer_text, confidence, is_answerable
        
        return answer_text if is_answerable else ""


# global model instance for reuse
_qa_system = None

def extractive_qa(question, context, model_name="deepset/roberta-base-squad2", 
                 max_length=512, return_confidence=False, device=None, confidence_threshold=None):
    global _qa_system
    
    # handle list context input
    if isinstance(context, list):
        if not context:
            return "" if not return_confidence else ("", 0.0, False)
        context = context[0]
    
    # init model if needed
    if _qa_system is None:
        _qa_system = ExtractiveQA(model_name=model_name, device=device)
    
    return _qa_system.extractive_qa(question, context, max_length, return_confidence, confidence_threshold)

def generative_qa_groq(query, contexts, model_name="llama-3.1-8b-instant"):
    """Generate answer using Groq API - FREE and FAST!"""
    if not contexts:
        return "No relevant context found in the document."
    
    context_text = "\n\n".join([
        c['text'] if isinstance(c, dict) else str(c) 
        for c in contexts[:3]  
    ])
    
    prompt = f"""Based on the following context from a document, answer the question. If the answer cannot be found in the context, respond with "No answer found in the document."

Context:
{context_text}

Question: {query}

Answer:"""
    
    try:
        # Get API key from environment or Streamlit secrets
        api_key = os.getenv('GROQ_API_KEY') or st.secrets.get('GROQ_API_KEY')

        if not api_key:
            print("Error: No API KEY")
        
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context. Be concise but comprehensive."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=300,
            top_p=0.9
        )
        
        answer = response.choices[0].message.content.strip()
        
        if answer.lower().startswith("answer:"):
            answer = answer[7:].strip()
        
        return answer
        
    except Exception as e:
        return f"Error with Groq API: {str(e)}"

def post_process_answer(answer: str, question: str) -> str:
    # handle empty answers
    if not answer:
        return ""
    
    # normalize unanswerable responses
    if answer.upper() in ["UNANSWERABLE", "NO ANSWER FOUND", "NOT FOUND"]:
        return ""
    
    # remove common gpt prefixes
    answer_lower = answer.lower()
    prefixes_to_remove = [
        "the answer is ",
        "according to the context, ",
        "based on the context, ",
        "answer: ",
    ]
    
    for prefix in prefixes_to_remove:
        if answer_lower.startswith(prefix):
            answer = answer[len(prefix):]
            break
    
    # clean up quotation marks
    if answer.startswith('"') and answer.endswith('"'):
        answer = answer[1:-1]
    
    # special handling for date questions
    question_lower = question.lower()
    if any(word in question_lower for word in ["when", "what year", "what century"]):
        import re
        year_match = re.search(r'\b(1\d{3}|20\d{2})\b', answer)
        century_match = re.search(r'\b(\d{1,2}(?:st|nd|rd|th)?\s*century)\b', answer, re.IGNORECASE)
        
        if year_match and len(year_match.group()) <= len(answer.split()[0]) * 2:
            return year_match.group()
        elif century_match:
            return century_match.group()
    
    # special handling for who questions
    elif question_lower.startswith("who"):
        if len(answer.split()) > 3:
            import re
            names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer)
            if names:
                longest_name = max(names, key=len)
                if len(longest_name.split()) <= 3:
                    return longest_name
    
    # special handling for what questions
    elif question_lower.startswith("what"):
        if len(answer.split()) > 5 and answer.endswith('.'):
            answer = answer.rstrip('.')
            import re
            match = re.search(r'(?:is|was|were|are)\s+([^,\.]+)', answer, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if len(extracted.split()) <= 3:
                    return extracted
    
    # final cleanup
    answer = answer.strip(' .,!?')
    
    # truncate very long answers
    if len(answer.split()) > 6:
        words = answer.split()
        for i in range(min(4, len(words))):
            if words[i].endswith(('.', ',', ';')):
                return ' '.join(words[:i+1]).rstrip('.,;')
        return ' '.join(words[:4])
    
    return answer

def main():
    # set up command line arguments
    parser = argparse.ArgumentParser(description="document-qa: retrieve + answer")
    parser.add_argument("--query",       type=str, required=True)
    parser.add_argument("--mode",        choices=["extractive", "generative"], default="extractive")
    parser.add_argument("--top_k",       type=int,   default=5)
    parser.add_argument("--index_path",  type=str,   default="faiss_index.index")
    parser.add_argument("--corpus_path", type=str,   default="corpus.pkl")
    parser.add_argument("--embedder",    type=str,   default="all-mpnet-base-v2")
    parser.add_argument("--gen_model",   type=str,   default="gpt-3.5-turbo")
    parser.add_argument("--min_score",   type=float, default=0.05)
    args = parser.parse_args()

    # load our search index and models
    index, corpus, embedder = load_resources(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        embedder_name=args.embedder
    )

    # find relevant document chunks
    hits = retrieve_and_rerank(
        query=args.query,
        index=index,
        corpus=corpus,
        embedder=embedder,
        faiss_k=20,
        top_k=args.top_k
    )
    contexts = [h["text"] for h in hits]

    # run qa based on selected method
    if args.mode == "extractive":
        answer = extractive_qa(
            question=args.query,
            context=contexts
        )
    else:
        answer = generative_qa_openai_optimized(
            query=args.query,
            contexts=contexts,
            model_name=args.gen_model
        )

    # show final answer
    print("=== answer ===")
    print(answer)


if __name__ == "__main__":
    main()