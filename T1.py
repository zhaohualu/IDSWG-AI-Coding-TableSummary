import streamlit as st
from docx import Document
import requests

# Extract demographic table from docx
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

# Send prompt to Ollama
def generate_summary_with_ollama(prompt, model="mistral"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# Streamlit app
st.set_page_config(page_title="Clinical Trial Demographics Summarizer", layout="centered")
st.title("Mistral-Powered Summary")

uploaded_file = st.file_uploader("Upload a .docx clinical trial file", type="docx")

if uploaded_file:
    with st.spinner("Extracting demographic data"):
        demographic_data = extract_demographic_lines(uploaded_file)
        if demographic_data:
            st.subheader("📋 Extracted Demographic Lines")
            st.write(demographic_data)

            if st.button("Summarize with Mistral (Ollama)"):
                with st.spinner("Querying Mistral via Ollama..."):
                    prompt = (
                        "You are a medical researcher. Summarize the key findings in bullet points from the following clinical trial demographic table:\n\n"
                        + "\n".join(demographic_data) + "\n\nSummary:"
                    )
                    summary = generate_summary_with_ollama(prompt)
                    st.subheader("🧠 Summary")
                    st.markdown(summary)
        else:
            st.warning("No relevant demographic data found.")