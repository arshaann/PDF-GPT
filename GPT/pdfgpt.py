import streamlit as st
import pdfplumber
from transformers import pipeline
import os

# Models ko initialize karna
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
summarizer = pipeline("summarization", model="t5-small")

# PDF se text extract karne ka function (chunk-wise for large files)
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text
                # Agar text bada ho jaye, toh break kar sakte hain (optional)
                if len(text) > 10_000_000:  # ~10MB text limit
                    st.warning("PDF too large, processing truncated.")
                    break
    except Exception as e:
        st.error(f"Error extracting text: {e}")
    return text

# Text ko chunks mein divide karna for summarization
def chunk_text(text, max_length=1000):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# PDF process karne ka function
def process_pdf(pdf_file, question=None):
    text = extract_text_from_pdf(pdf_file)
    
    # Summary ke liye chunks
    chunks = chunk_text(text)
    summary = ""
    for chunk in chunks[:5]:  # Sirf pehle 5 chunks summarize karo (optimization)
        summary += summarizer(chunk, max_length=50, min_length=20, do_sample=False)[0]['summary_text'] + " "
    
    # Question answering (full text pe)
    answer = qa_model(question=question, context=text[:10_000])['answer'] if question else "No question provided."
    
    return {"summary": summary.strip(), "answer": answer}

# Streamlit UI
st.title("PDFGPT - Analyze Your PDFs with AI")
st.write("Upload a PDF file (max 200 MB), ask a question (optional), and get a summary and answer!")

# PDF upload with 200 MB limit
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)

# File size check
if uploaded_file:
    file_size = uploaded_file.size / (1024 * 1024)  # Size in MB
    if file_size > 200:
        st.error("File size exceeds 200 MB limit. Please upload a smaller file.")
        uploaded_file = None
    else:
        st.write(f"File size: {file_size:.2f} MB")

# Question input
question = st.text_input("Ask a question about the PDF (optional)", "")

# Process button
if st.button("Process PDF") and uploaded_file:
    with st.spinner("Processing large PDF... This may take a moment."):
        result = process_pdf(uploaded_file, question)
        st.subheader("Summary")
        st.write(result["summary"])
        st.subheader("Answer")
        st.write(result["answer"])

# Instructions
st.write("Note: Upload a PDF (up to 200 MB) and click 'Process PDF' to see the results.")