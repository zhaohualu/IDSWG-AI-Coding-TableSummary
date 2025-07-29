import streamlit as st
from docx import Document
import requests
import os
import tempfile
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import faiss
from pdf2docx import Converter

# -----------------------------
# THEME & STYLING
# -----------------------------
st.set_page_config(page_title="Summary Wizard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for dark theme, green accents, monospace font, and larger headings
st.markdown(
    """
    <style>
    body, .stApp { background-color: #101c14 !important; color: #e0ffe0 !important; font-family: 'Fira Mono', 'Menlo', 'Monaco', monospace !important; }
    .stSidebar { background-color: #14281d !important; }
    .stButton>button { background-color: #1e4023 !important; color: #e0ffe0 !important; border-radius: 8px; }
    .stCard { background: #1a2e22 !important; border: 1px solid #2e4d36 !important; border-radius: 12px; }
    h1, h2, h3, h4 { font-size: 2.2em !important; color: #6aff6a !important; font-family: 'Fira Mono', monospace !important; }
    .stChatMessage { background: #1a2e22 !important; border-radius: 10px; }
    .stTable { background: #1a2e22 !important; }
    .stMarkdown { color: #e0ffe0 !important; }
    .sidebar-nav-btn {
        background-color: transparent;
        color: #e0ffe0;
        border: none;
        border-radius: 8px;
        padding: 0.5em 1em;
        margin-bottom: 0.3em;
        width: 100%;
        text-align: left;
        font-size: 1.1em;
        font-family: 'Fira Mono', monospace;
        cursor: pointer;
        transition: background 0.2s, color 0.2s;
    }
    .sidebar-nav-btn.selected {
        background-color: #2e4d36 !important;
        color: #6aff6a !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# LOGO & SIDEBAR
# -----------------------------
PAGES = [
    ("Home", "🏠"),
    ("Upload & Summarize", "⬆️"),
    ("AI Agent", "🤖"),
    ("History", "📜"),
]

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

with st.sidebar:
    st.image("https://i.imgur.com/0y8Ftya.png", width=80)
    st.markdown("<h2 style='margin-bottom:0;'>Summary Wizard</h2>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1em; color:#b0ffb0;'>Welcome, user</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<span style='font-size:0.9em; color:#6aff6a;'>Navigate using the sidebar</span>", unsafe_allow_html=True)

    for name, icon in PAGES:
        selected = st.session_state["page"] == name
        button = st.button(f"{icon} {name}", key=f"nav-{name}")
        if button:
            st.session_state["page"] = name
        # Custom CSS for selected button
        if selected:
            st.markdown(
                f"""
                <style>
                div[data-testid="stSidebar"] button[data-testid="baseButton"][key="nav-{name}"] {{
                    background-color: #2e4d36 !important;
                    color: #6aff6a !important;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

page = st.session_state["page"]

# -----------------------------
# 1. SETUP: Embedding & Storage
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = 384
faiss_index = faiss.IndexFlatL2(embedding_dim)
doc_id_map = []

mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["clinical_memory"]
collection = db["summaries"]

# Bootstrap a dummy memory if database is empty
def bootstrap_memory():
    if collection.count_documents({}) == 0:
        lines = ["Age | 60", "Female | 70%"]
        summary = "Most patients were women around 60 years old."
        text = " ".join(lines)
        embedding = embedder.encode([text])
        doc = {
            "lines": lines,
            "summary": summary,
            "user": "bootstrap",
            "tags": ["test"],
            "timestamp": datetime.utcnow()
        }
        result = collection.insert_one(doc)
        doc_id_map.append(result.inserted_id)
        faiss_index.add(np.array(embedding))

bootstrap_memory()

def rebuild_faiss_and_docidmap():
    global doc_id_map, faiss_index
    doc_id_map = []
    faiss_index.reset()
    all_docs = list(collection.find())
    if all_docs:
        texts = [" ".join(doc.get("lines", [])) for doc in all_docs]
        embeddings = embedder.encode(texts)
        faiss_index.add(np.array(embeddings))
        doc_id_map = [doc["_id"] for doc in all_docs]

# Call this after MongoDB setup and before using memory functions
rebuild_faiss_and_docidmap()

# -----------------------------

# -----------------------------
# 2. MEMORY FUNCTIONS (unchanged)
# -----------------------------
def store_table_memory(table_lines, summary, user="anon", tags=[]):
    text = " ".join(table_lines)
    embedding = embedder.encode([text])
    doc = {
        "lines": table_lines,
        "summary": summary,
        "user": user,
        "tags": tags,
        "timestamp": datetime.utcnow()
    }
    result = collection.insert_one(doc)
    doc_id_map.append(result.inserted_id)
    faiss_index.add(np.array(embedding))

def retrieve_similar_memories(new_table_lines, top_k=3):
    query_text = " ".join(new_table_lines)
    query_embedding = embedder.encode([query_text])
    D, I = faiss_index.search(np.array(query_embedding), top_k)
    results = []
    for idx in I[0]:
        if idx < len(doc_id_map):
            mongo_id = doc_id_map[idx]
            doc = collection.find_one({"_id": mongo_id})
            results.append(doc)
    return results

# -----------------------------
# 3. DOCX TABLE EXTRACTOR (unchanged)
# -----------------------------
def extract_demographic_lines(docx_file):
    doc = Document(docx_file)
    tables = doc.tables
    demographics_lines = []
    for table in tables:
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_data:
                line = " | ".join(row_data)
                if any(keyword in line.lower() for keyword in ["age", "sex", "race", "ecog", "smoke", "therapy", "mutation", "metastases"]):
                    demographics_lines.append(line)
    return demographics_lines

# -----------------------------
# 4. PDF TO DOCX CONVERTER (unchanged)
# -----------------------------
def convert_pdf_to_docx(pdf_path, output_path):
    cv = Converter(pdf_path)
    cv.convert(output_path, start=0, end=None)
    cv.close()
    return output_path

# -----------------------------
# 5. OLLAMA QUERY FUNCTION (unchanged)
# -----------------------------
def generate_summary_with_ollama(prompt, model="mistral"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

# -----------------------------
# 6. MULTI-PAGE NAVIGATION
# -----------------------------
# PAGES = ["Home", "Upload & Summarize", "AI Agent", "History"]
# page = st.sidebar.radio("Go to", PAGES, index=0)

# -----------------------------
# 7. PAGE LOGIC
# -----------------------------
if page == "Home":
    # Popup instructions every visit
    if st.session_state.get("show_instructions", True):
        st.info("""**How to use Summary Wizard:**\n\n1. Upload a clinical trial or research document.\n2. Get an AI-powered summary.\n3. Chat with the AI Agent about your document.\n4. View your history anytime!""")
        if st.button("Got it!"):
            st.session_state["show_instructions"] = False
    st.header("Welcome to Summary Wizard")
    st.write("Upload your oncology research or statistics and let the AI do the heavy lifting.")
    # Recent summaries (side list)
    st.subheader("Recent Summaries")
    recent = collection.find().sort("timestamp", -1).limit(5)
    with st.container():
        for doc in recent:
            st.markdown(f"- **{doc.get('summary', '')[:40]}...**")

elif page == "Upload & Summarize":
    st.header("Upload & Summarize")
    uploaded_file = st.file_uploader("Drag and drop or click to upload a .pdf or .docx clinical trial file", type=["pdf", "docx"])
    if uploaded_file:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            # Convert PDF to DOCX if needed
            if uploaded_file.name.endswith(".pdf"):
                st.info("Converting PDF to DOCX...")
                docx_path = os.path.join(tmpdir, "converted.docx")
                convert_pdf_to_docx(file_path, docx_path)
            else:
                docx_path = file_path
            with st.spinner("Extracting demographic data..."):
                demographic_lines = extract_demographic_lines(docx_path)
                if demographic_lines:
                    st.subheader("📋 Extracted Demographic Lines")
                    st.write(demographic_lines)
                    if st.button("Summarize with Mistral + Memory"):
                        with st.spinner("Retrieving memory and summarizing..."):
                            memory_summaries = retrieve_similar_memories(demographic_lines, top_k=2)
                            context = "\n\n".join(["Past similar summary: " + m["summary"] for m in memory_summaries])
                            prompt = (
                                "You are a medical researcher. Use the following past clinical trial summaries to help interpret a new demographic table. Write summary in bullet points and keep it concise but detailed."
                                f"\n\n{context}\n\nNow summarize this table:\n"
                                + "\n".join(demographic_lines) + "\n\nSummary:"
                            )
                            summary = generate_summary_with_ollama(prompt)
                            st.subheader("🧠 Summary")
                            st.markdown(f"<div class='stCard'>{summary}</div>", unsafe_allow_html=True)
                            store_table_memory(demographic_lines, summary)
                else:
                    st.warning("No relevant demographic data found.")

elif page == "AI Agent":
    st.header("AI Agent Q&A")
    st.write("Ask questions about your uploaded document or compare with past documents.")
    uploaded_file = st.file_uploader("Upload a new document to chat about (optional)", type=["pdf", "docx"], key="chat_upload")
    if uploaded_file:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            if uploaded_file.name.endswith(".pdf"):
                st.info("Converting PDF to DOCX...")
                docx_path = os.path.join(tmpdir, "converted.docx")
                convert_pdf_to_docx(file_path, docx_path)
            else:
                docx_path = file_path
            demographic_lines = extract_demographic_lines(docx_path)
            st.session_state["chat_doc_lines"] = demographic_lines
    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    user_input = st.text_input("Ask the AI Agent a question:")
    if user_input:
        doc_lines = st.session_state.get("chat_doc_lines", [])
        prompt = (
            "You are an expert medical AI. Use the following document lines to answer the user's question.\n"
            + "\n".join(doc_lines) + f"\n\nQuestion: {user_input}\nAnswer:"
        )
        answer = generate_summary_with_ollama(prompt)
        st.session_state["chat_history"].append((user_input, answer))
    st.subheader("Chat History")
    for q, a in st.session_state["chat_history"]:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**AI:** {a}")

elif page == "History":
    st.header("History")
    st.write("View your past uploads, summaries, and Q&A.")
    search = st.text_input("Search history:")
    # Fetch all docs
    docs = list(collection.find())
    # Filter by search
    if search:
        docs = [d for d in docs if search.lower() in (d.get("summary", "") + " ".join(d.get("lines", []))).lower()]
    # Show as table
    import pandas as pd
    if docs:
        df = pd.DataFrame([
            {
                "Date": d["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "Summary": d.get("summary", "")[:60],
                "Lines": ", ".join(d.get("lines", [])[:2]) + ("..." if len(d.get("lines", [])) > 2 else "")
            }
            for d in docs
        ])
        st.dataframe(df)
    else:
        st.info("No history found.")
