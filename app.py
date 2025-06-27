# app.py
import streamlit as st
import tempfile
import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer
import groq
from groq import Groq
from src.qa import extractive_qa  # My existing extractive QA function
import time
import hashlib
from collections import defaultdict

# Configure page
st.set_page_config(
    page_title="PDF Question Answering System",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global dark theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.3), 0 0 80px rgba(139, 92, 246, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255, 255, 255, 0.1) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        margin: 1rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem 2rem;
        border-left: 4px solid #6366f1;
        margin: 2.5rem 0 2rem 0;
        border-radius: 0 16px 16px 0;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        border: 1px solid #374151;
    }
    
    .section-header h2 {
        margin: 0;
        color: #f1f5f9;
        font-size: 1.4rem;
        font-weight: 600;
    }
    
    /* Custom message boxes */
    .custom-success {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        border: 1px solid #10b981;
        color: #d1fae5;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border-left: 4px solid #10b981;
        font-weight: 500;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2);
    }
    
    .custom-warning {
        background: linear-gradient(135deg, #92400e 0%, #b45309 100%);
        border: 1px solid #f59e0b;
        color: #fef3c7;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border-left: 4px solid #f59e0b;
        font-weight: 500;
        box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2);
    }
    
    .custom-info {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        border: 1px solid #60a5fa;
        color: #dbeafe;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border-left: 4px solid #60a5fa;
        font-weight: 500;
        box-shadow: 0 8px 20px rgba(96, 165, 250, 0.2);
    }
    
    /* Answer boxes - Make them really stand out */
    .answer-box {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        border: 2px solid #4b5563;
        border-radius: 16px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.05);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .answer-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, transparent 0%, rgba(255, 255, 255, 0.03) 50%, transparent 100%);
        pointer-events: none;
    }
    
    .answer-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.1);
    }
    
    .answer-box.extractive {
        border-left: 4px solid #10b981;
        background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%);
        box-shadow: 0 20px 40px rgba(16, 185, 129, 0.2), 0 0 60px rgba(16, 185, 129, 0.1);
    }
    
    .answer-box.extractive:hover {
        box-shadow: 0 25px 50px rgba(16, 185, 129, 0.3), 0 0 80px rgba(16, 185, 129, 0.15);
    }
    
    .answer-box.generative {
        border-left: 4px solid #3b82f6;
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #2563eb 100%);
        box-shadow: 0 20px 40px rgba(59, 130, 246, 0.2), 0 0 60px rgba(59, 130, 246, 0.1);
    }
    
    .answer-box.generative:hover {
        box-shadow: 0 25px 50px rgba(59, 130, 246, 0.3), 0 0 80px rgba(59, 130, 246, 0.15);
    }
    
    .answer-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #f8fafc;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .answer-subtitle {
        font-size: 0.9rem;
        color: #cbd5e1;
        margin-bottom: 1.5rem;
        font-weight: 400;
        opacity: 0.8;
    }
    
    .answer-content {
        font-size: 1rem;
        line-height: 1.8;
        color: #f1f5f9;
        font-weight: 400;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    .answer-content strong {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.4);
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #c084fc 100%);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border-radius: 12px;
        border: 2px dashed #6366f1;
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        color: #e2e8f0;
    }
    
    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 1.8rem;
        border-radius: 12px;
        border: 1px solid #4b5563;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #60a5fa;
        margin-bottom: 0.3rem;
        text-shadow: 0 2px 4px rgba(96, 165, 250, 0.3);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #374151;
        padding: 1rem 1.5rem;
        font-size: 1rem;
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        color: #f1f5f9;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2);
        background: #1f2937;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #9ca3af;
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #374151;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        border-radius: 10px;
        font-weight: 600;
        color: #f1f5f9;
        border: 1px solid #4b5563;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
        transform: translateY(-1px);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        border: 1px solid #4b5563;
        border-radius: 8px;
        color: #e2e8f0;
        font-family: 'Inter', monospace;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }
    
    /* Sidebar text styling */
    .css-1d391kg .markdown-text-container {
        color: #e2e8f0;
    }
    
    .css-1d391kg h3 {
        color: #f1f5f9;
        font-weight: 600;
    }
    
    /* Success/warning in sidebar */
    .css-1d391kg .element-container .stSuccess {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        color: #d1fae5;
        border-radius: 8px;
    }
    
    .css-1d391kg .element-container .stWarning {
        background: linear-gradient(135deg, #92400e 0%, #b45309 100%);
        color: #fef3c7;
        border-radius: 8px;
    }
    
    /* Column styling for better separation */
    .css-1kyxreq {
        gap: 2rem;
    }
    
    /* Override Streamlit's default text colors */
    .css-1d391kg p, .css-1d391kg li {
        color: #cbd5e1 !important;
    }
    
    /* File uploader text */
    .css-1cpxqw2 {
        color: #e2e8f0 !important;
    }
    
    /* Better scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1f2937;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4b5563;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #6b7280;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_pdf' not in st.session_state:
    st.session_state.processed_pdf = False
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None

if 'user_requests' not in st.session_state:
    st.session_state.user_requests = defaultdict(list)
if 'total_requests_today' not in st.session_state:
    st.session_state.total_requests_today = 0

def get_user_identifier():
    """Create a simple user identifier for rate limiting."""
    # Use session state ID as a simple identifier
    if 'user_id' not in st.session_state:
        # Create a simple hash based on session info
        session_info = str(st.session_state) + str(time.time())
        st.session_state.user_id = hashlib.md5(session_info.encode()).hexdigest()[:8]
    return st.session_state.user_id

def check_rate_limit(user_id, max_requests=3, time_window=300):  # 3 requests per 5 minutes
    """Check if user has exceeded rate limit."""
    now = time.time()
    user_requests = st.session_state.user_requests[user_id]
    
    # Remove old requests outside the time window
    user_requests[:] = [req_time for req_time in user_requests if now - req_time < time_window]
    
    if len(user_requests) >= max_requests:
        return False, max_requests - len(user_requests), time_window
    
    return True, max_requests - len(user_requests) - 1, time_window

def record_request(user_id):
    """Record a new request for the user."""
    st.session_state.user_requests[user_id].append(time.time())
    st.session_state.total_requests_today += 1

def format_time_remaining(seconds):
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        return f"{int(seconds/60)} minutes"
    else:
        return f"{int(seconds/3600)} hours"

@st.cache_resource
def load_models():
    """Load and cache the models to avoid reloading."""
    embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    tokenizer = embedder.tokenizer
    return embedder, reranker, tokenizer

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        doc = fitz.open(tmp_path)
        pages = [page.get_text() for page in doc]
        doc.close()
        full_text = "\n".join(pages)
        return full_text
    finally:
        os.unlink(tmp_path)

def chunk_text(text, tokenizer, chunk_size=512, overlap=256):
    """Chunk text into overlapping segments."""
    if not text.strip():
        return []
    
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    if len(token_ids) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(token_ids):
        end = start + chunk_size
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids)
        chunks.append(chunk_text)
        
        if end >= len(token_ids):
            break
        start += chunk_size - overlap
    
    return chunks

def build_search_index(chunks, embedder):
    """Build FAISS search index from text chunks."""
    if not chunks:
        return None, []
    
    embeddings = embedder.encode(
        chunks, 
        show_progress_bar=False,
        convert_to_numpy=True
    )
    
    faiss.normalize_L2(embeddings)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    return index, chunks

def retrieve_and_rerank(query, index, chunks, embedder, reranker, faiss_k=10, top_k=5):
    """Retrieve and rerank relevant chunks."""
    if index is None or not chunks:
        return []
    
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    
    k = min(faiss_k, index.ntotal)
    scores, ids = index.search(q_emb, k)
    
    candidates = []
    for i, (score, idx) in enumerate(zip(scores[0], ids[0])):
        if idx >= 0 and idx < len(chunks): 
            candidates.append({
                'text': chunks[idx],
                'faiss_score': float(score),
                'chunk_id': int(idx)
            })
    
    if not candidates:
        return []
    
    pairs = [[query, c['text']] for c in candidates]
    rerank_scores = reranker.predict(pairs)
    
    for c, score in zip(candidates, rerank_scores):
        c['rerank_score'] = float(score)
    
    candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
    return candidates[:top_k]

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

def main():
    st.markdown("""
    <div class="main-header">
        <h1>PDF Question Answering System</h1>
        <p>Upload any PDF document and get intelligent answers to your questions using advanced AI models</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Loading AI models..."):
        embedder, reranker, tokenizer = load_models()
    
    st.markdown('<div class="section-header"><h2>Step 1: Upload Your Document</h2></div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Select a PDF file to analyze", 
        type="pdf",
        help="Choose any PDF document containing text content"
    )
    
    if uploaded_file is not None:
        if st.session_state.pdf_name != uploaded_file.name:
            st.session_state.processed_pdf = False
            st.session_state.pdf_name = uploaded_file.name
        
        st.markdown(f'<div class="custom-success">Document loaded: <strong>{uploaded_file.name}</strong></div>', unsafe_allow_html=True)
        
        if not st.session_state.processed_pdf:
            with st.spinner("Processing document..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
                
                if not pdf_text.strip():
                    st.markdown('<div class="custom-warning">No readable text found in this PDF. Please ensure the document contains extractable text.</div>', unsafe_allow_html=True)
                    return
                
                chunks = chunk_text(pdf_text, tokenizer, chunk_size=512, overlap=256)
                
                if not chunks:
                    st.markdown('<div class="custom-warning">Unable to process the text from this document.</div>', unsafe_allow_html=True)
                    return
                
                index, processed_chunks = build_search_index(chunks, embedder)
                
                st.session_state.index = index
                st.session_state.chunks = processed_chunks
                st.session_state.processed_pdf = True
                
                st.markdown(f'<div class="custom-success">Processing complete! Document has been divided into {len(chunks)} searchable sections.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="custom-info">Document ready for questions.</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-header"><h2>Step 2: Ask Your Question</h2></div>', unsafe_allow_html=True)
        
        question = st.text_input(
            "What would you like to know about this document?",
            placeholder="Example: What are the main findings discussed in this paper?"
        )
        
        if question and st.session_state.processed_pdf:
            user_id = get_user_identifier()
            can_proceed, remaining_requests, time_window = check_rate_limit(user_id)

            if not can_proceed:
                user_requests = st.session_state.user_requests[user_id]
                if user_requests:
                    oldest_request = min(user_requests)
                    time_until_reset = time_window - (time.time() - oldest_request)
                    time_str = format_time_remaining(time_until_reset)
                else:
                    time_str = format_time_remaining(time_window)
                
                st.markdown(f'''
                <div class="rate-limit-warning">
                    <strong>Rate Limit Exceeded</strong><br>
                    You've reached the maximum of 3 questions per 5 minutes to prevent API abuse.<br>
                    Please wait <strong>{time_str}</strong> before asking another question.<br>
                    <small>This helps keep the service free for everyone! 🙏</small>
                </div>
                ''', unsafe_allow_html=True)
                return
            
            if remaining_requests >= 0:
                if remaining_requests > 0:
                    st.markdown(f'<div class="custom-info">You have <strong>{remaining_requests}</strong> questions remaining in the next 5 minutes.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="custom-warning">This is your last question for the next 5 minutes.</div>', unsafe_allow_html=True)
            
            record_request(user_id)

            with st.spinner("Searching through document..."):
                relevant_chunks = retrieve_and_rerank(
                    question, 
                    st.session_state.index, 
                    st.session_state.chunks, 
                    embedder, 
                    reranker,
                    faiss_k=10,
                    top_k=5
                )
            
            if not relevant_chunks:
                st.markdown('<div class="custom-warning">No relevant content found for your question in this document.</div>', unsafe_allow_html=True)
                return
            
            st.markdown('<div class="section-header"><h2>Step 3: AI Analysis Results</h2></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="answer-box extractive">
                    <div class="answer-title">Extractive Analysis</div>
                    <div class="answer-subtitle">Precise text extraction using RoBERTa model</div>
                """, unsafe_allow_html=True)
                
                with st.spinner("Analyzing document..."):
                    context_text = relevant_chunks[0]['text']
                    extractive_answer, confidence, is_answerable = extractive_qa(
                        question, 
                        context_text, 
                        return_confidence=True,
                        confidence_threshold=0.4
                    )
                
                if extractive_answer:
                    st.markdown(f'<div class="answer-content"><strong>Answer:</strong> {extractive_answer}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-container"><div class="metric-value">{confidence:.3f}</div><div class="metric-label">Confidence Score</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="custom-warning">No specific answer found in the text.</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-container"><div class="metric-value">{confidence:.3f}</div><div class="metric-label">Confidence (Below Threshold)</div></div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="answer-box generative">
                    <div class="answer-title">Generative Analysis</div>
                    <div class="answer-subtitle">Comprehensive response using Groq Llama model</div>
                """, unsafe_allow_html=True)
                
                with st.spinner("Generating comprehensive response..."):
                    generative_answer = generative_qa_groq(question, relevant_chunks)
                
                if generative_answer and "No answer found" not in generative_answer:
                    st.markdown(f'<div class="answer-content"><strong>Answer:</strong> {generative_answer}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="custom-warning">Unable to generate a comprehensive answer from the available content.</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with st.expander("View Source Context", expanded=False):
                st.subheader("Most Relevant Document Sections")
                for i, chunk in enumerate(relevant_chunks[:3]):
                    st.markdown(f"**Section {i+1}** (Relevance Score: {chunk['rerank_score']:.3f})")
                    st.text_area(
                        f"Content {i+1}", 
                        chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'],
                        height=100,
                        key=f"context_{i}"
                    )
    
    else:
        st.markdown('<div class="custom-info">Please upload a PDF document to begin the analysis.</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown("### System Information")
    st.sidebar.markdown("""
    This application uses two complementary AI approaches to answer questions about your documents:
    
    **Extractive Analysis**
    - Identifies exact text passages that answer your question
    - Provides confidence scores for reliability assessment
    - Best for finding specific facts and details
    
    **Generative Analysis**
    - Creates comprehensive answers by synthesizing information
    - Handles complex questions requiring reasoning
    - Provides more natural, conversational responses
    
    **Technical Features**
    - Advanced text processing and chunking
    - Semantic search using FAISS indexing
    - Cross-encoder reranking for relevance
    - Confidence scoring and threshold filtering
    """)
    
    st.sidebar.markdown("### API Configuration")
    if not os.getenv('OPENAI_API_KEY'):
        st.sidebar.warning("OpenAI API key required for generative analysis. Set OPENAI_API_KEY environment variable.")
    else:
        st.sidebar.success("Groq API configured successfully")

if __name__ == "__main__":
    main()