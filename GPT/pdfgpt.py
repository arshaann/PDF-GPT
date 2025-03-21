import streamlit as st
import pdfplumber
from transformers import pipeline
import re
from difflib import SequenceMatcher

# Models ko initialize karna
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
summarizer = pipeline("summarization", model="t5-small")

# Function to check similarity between two texts
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Function to remove email addresses from text
def remove_emails(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.sub(email_pattern, '[Email Removed]', text)

# PDF se text extract karne ka function
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text
                if len(text) > 10_000_000:
                    st.warning("PDF too large, processing truncated.")
                    break
    except Exception as e:
        st.error(f"Error extracting text: {e}")
    return text

# Text ko chunks mein divide karna
def chunk_text(text, max_length=1000):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# PDF process karne ka function
def process_pdf(pdf_file, question=None, summary_limit=500, answer_limit=150):
    text = extract_text_from_pdf(pdf_file)
    
    # Summary ke liye chunks
    chunks = chunk_text(text)
    summary = ""
    for chunk in chunks[:10]:
        # Adjust max_length per chunk dynamically based on desired summary_limit
        chunk_max_length = max(50, summary_limit // 10)  # Distribute limit across chunks
        summary += summarizer(chunk, max_length=chunk_max_length, min_length=40, do_sample=False)[0]['summary_text'] + " "
    
    # Trim summary to user-defined word limit
    summary_words = summary.split()
    if len(summary_words) > summary_limit:
        summary = " ".join(summary_words[:summary_limit]) + "..."
    
    # Question answering
    answer = qa_model(question=question, context=text[:10_000])['answer'] if question else "No question provided."
    
    # Check if answer is too similar to summary
    if similarity(answer.lower(), summary.lower()) > 0.8:
        answer = "This question might be too broad. Please ask a more specific question (e.g., 'What is the degree mentioned in the PDF?')."
    
    # Post-process the answer: Trim to user-defined answer_limit
    answer_words = answer.split()
    if len(answer_words) > answer_limit:
        answer = " ".join(answer_words[:answer_limit]) + "... (Answer truncated, please rephrase the question for better results)"
    
    # Sanitize outputs
    summary = remove_emails(summary)
    answer = remove_emails(answer)
    
    return {"summary": summary.strip(), "answer": answer}

# Streamlit UI
st.title("PDFGPT - Analyze Your PDFs (Up to 200 MB)")
st.write("Upload a PDF file (max 200 MB), ask a question (optional), and get a summary and answer!")

# Warning for sensitive info
st.warning("Please ensure your PDF does not contain sensitive information like email addresses or personal details.")

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

# Summary word limit slider
summary_limit = st.slider("Select summary word limit", 100, 1000, 500)  # Range: 100 to 1000, default: 500

# Answer word limit slider
answer_limit = st.slider("Select answer word limit", 50, 300, 150)  # Range: 50 to 300, default: 150

# Question input
question = st.text_input("Ask a question about the PDF (optional)", "")

# Process button
if st.button("Process PDF") and uploaded_file:
    with st.spinner("Processing large PDF... This may take a moment."):
        result = process_pdf(uploaded_file, question, summary_limit, answer_limit)  # Pass both limits
        st.subheader("Summary")
        st.write(result["summary"])
        st.subheader("Answer")
        st.write(result["answer"])

# Instructions
st.write("Note: Upload a PDF (up to 200 MB) and click 'Process PDF' to see the results.")
