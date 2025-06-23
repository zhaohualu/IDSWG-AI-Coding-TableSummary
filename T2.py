import streamlit as st
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load sentence transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_lines_from_tables(docx_file):
    doc = Document(docx_file)
    lines = []
    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_text:
                lines.append(" | ".join(row_text))
    return lines

def summarize_lines(lines, model, top_k=5):
    embeddings = model.encode(lines)
    mean_embedding = np.mean(embeddings, axis=0)
    sims = cosine_similarity([mean_embedding], embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    summary = [lines[i] for i in top_indices]
    return summary

# UI
st.set_page_config(page_title="Clinical Table Summarizer", layout="centered")
st.title("🧬 Clinical Trial Table Summary using Sentence Transformers")

uploaded_file = st.file_uploader("Upload your .docx file", type="docx")

if uploaded_file:
    with st.spinner("Extracting rows..."):
        lines = extract_lines_from_tables(uploaded_file)
        st.subheader("📋 Extracted Table Rows")
        st.write(lines)

    if st.button("Generate Summary"):
        model = load_model()
        summary = summarize_lines(lines, model, top_k=5)

        st.subheader("🧠 Summary of Key Findings")
        for i, sent in enumerate(summary, 1):
            st.write(f"**{i}.** {sent}")