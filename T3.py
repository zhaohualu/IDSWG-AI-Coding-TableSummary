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
# 1. SETUP: Embedding & Storage
# -----------------------------

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = 384
faiss_index = faiss.IndexFlatL2(embedding_dim)
doc_id_map = []

mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["clinical_memory"]
collection = db["summaries"]


# -----------------------------
# 2. MEMORY FUNCTIONS
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
        mongo_id = doc_id_map[idx]
        doc = collection.find_one({"_id": mongo_id})
        results.append(doc)
    return results


# -----------------------------
# 3. DOCX TABLE EXTRACTOR
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
# 4. PDF TO DOCX CONVERTER
# -----------------------------
def convert_pdf_to_docx(pdf_path, output_path):
    cv = Converter(pdf_path)
    cv.convert(output_path, start=0, end=None)
    cv.close()
    return output_path

# -----------------------------
# 5. OLLAMA QUERY FUNCTION
# -----------------------------
def generate_summary_with_ollama(prompt, model="mistral"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]


# -----------------------------
# 6. STREAMLIT APP
# -----------------------------
st.set_page_config(page_title="Clinical Trial Demographics Summarizer", layout="centered")
st.title("🧬 Mistral-Powered Summary (via Ollama) with Memory")

uploaded_file = st.file_uploader("Upload a .pdf or .docx clinical trial file", type=["pdf", "docx"])

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
                            "You are a medical researcher. Use the following past clinical trial summaries to help interpret a new demographic table."
                            f"\n\n{context}\n\nNow summarize this table:\n"
                            + "\n".join(demographic_lines) + "\n\nSummary:"
                        )

                        summary = generate_summary_with_ollama(prompt)
                        st.subheader("🧠 Summary")
                        st.markdown(summary)

                        store_table_memory(demographic_lines, summary)
            else:
                st.warning("No relevant demographic data found.")
