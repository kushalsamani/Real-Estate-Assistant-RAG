# Real Estate Assistant – RAG-Powered Research App

This is a lightweight RAG (Retrieval-Augmented Generation) application built with **LangChain**, **Chroma**, and **Streamlit**. It lets you input URLs of real estate articles (or any webpage), scrapes the content, stores it in a vector database, and answers questions using **Groq-hosted LLaMA 3**.

---

## Features

- Input up to 3 webpage URLs
- Web scraping using `WebBaseLoader` (LangChain)
- Text chunking and embedding with HuggingFace's `Qwen` model
- Vector storage using Chroma
- LLM question answering with LLaMA 3 (via Groq)
- Simple frontend powered by Streamlit

---

## Tech Stack

- Python 3.10+
- LangChain
- HuggingFace Embeddings
- Chroma VectorDB
- Groq LLM (LLaMA 3)
- Streamlit

---

## Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/kushalsamani/Real-Estate-Assistant-RAG.git
cd Real-Estate-Assistant-RAG
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create `.env` File

Create a `.env` file in the root directory with your Groq API key:

```env
GROQ_API_KEY=your_key_here
```

---

## Run the App

```bash
streamlit run main.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Example Use Case

1. Enter URLs like:
   - https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html
   - https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html

2. Ask:

> “What is the current 30-year fixed mortgage rate and the date?”

The app will extract, embed, and generate an answer using retrieved context + LLM.


---

## Contact

**Kushal Samani**  
- Email: kushalsamani04@gmail.com  
- LinkedIn: https://www.linkedin.com/in/kushalsamani
