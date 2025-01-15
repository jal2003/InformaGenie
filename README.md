# **InformaGenie 🧞‍♂️ | Your Personal Information Scraping AI**

InformaGenie isn’t just another project—it’s a problem solver born out of frustration. Ever spent hours researching across countless websites, copying and pasting bits of information, only to end up with a cluttered mess of tabs and a headache? That’s where InformaGenie comes in.

This AI-powered agent fetches the data *for you* from across the web and gives you precise answers in seconds. No more endless scrolling or skimming—just ask your question, and let InformaGenie do the magic.

---

## **Why I Built InformaGenie**

I often found myself wasting time digging through long paragraphs on multiple sites, hunting for a specific answer. I thought, *“There’s got to be a better way!”* That idea led me to build InformaGenie, a tool designed to simplify the information-gathering process using cutting-edge AI technologies.

---

## **How It Works**

InformaGenie leverages the power of **Python**, **Ollama**, **LangChain**, and **Retrieval-Augmented Generation (RAG)** to provide highly accurate answers. Here’s a breakdown of the tech stack:

- **Python**: The backbone of the application, handling the core logic and web scraping.
- **Ollama**: AI model for natural language understanding and processing.
- **LangChain**: Manages LLM-driven workflows and improves response generation.
- **RAG (Retrieval-Augmented Generation)**: Ensures the AI gives factually grounded answers by retrieving relevant web data before generating a response.
- **Streamlit**: Provides a clean and interactive UI, making the app easy and fun to use.

---

## **Features**

- 🔍 **AI-Powered Web Scraper**: Extracts relevant data from multiple sources in real-time.
- 🤖 **Ask Anything**: Input any question and get a concise, accurate answer in seconds.
- 🎯 **High Accuracy**: Uses RAG to combine web retrieval with LLM capabilities for precise answers.
- 🚀 **Simple & Clean UI**: Built with Streamlit, offering an intuitive user experience.

---

## **Tech Stack**

| **Technology**         | **Purpose**                                  |
|------------------------|----------------------------------------------|
| Python                 | Core logic and data processing               |
| Ollama                 | Language model for NLP                       |
| LangChain              | LLM-driven workflows and RAG implementation  |
| Streamlit              | UI framework for the web app                 |
| Retrieval-Augmented Generation (RAG) | Accurate information retrieval |

---

## **Getting Started**

Follow these steps to get InformaGenie up and running on your local machine:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jal2003/informagenie.git
   cd informagenie
   python -m venv env
   source env/bin/activate   # On Windows, use `env\Scripts\activate`
   pip install -r requirements.txt
   streamlit run app.py
   

## **Future Enhancements**

🌐 Support for additional websites: Expanding the range of web sources.
🧠 Better contextual understanding: Fine-tuning the LLM for more nuanced responses.
📊 Data visualization: Displaying results in graphs and tables for richer insights.

