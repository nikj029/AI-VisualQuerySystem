# AI-Powered Diagram QA System

This project is a **Streamlit-based application** that uses **Google Gemini** and **Neo4j Graph Database** to extract relationships from architecture or data flow diagrams and answer user queries with RAG (Retrieval-Augmented Generation).

## Features

- Upload architecture diagrams (`.png`, `.jpg`, `.jpeg`)
- Extract relationships using **Google Gemini (Vision + LLM)**
- Store extracted entities and relationships in **Neo4j**
- Perform **semantic search** using TF-IDF
- Ask questions and get smart responses using **contextual RAG**
- Simple and beautiful **Streamlit UI**

---

## Tech Stack

- **Frontend/UI:** Streamlit
- **AI/LLM:** Google Gemini API
- **Graph DB:** Neo4j (via Py2Neo)
- **ML/NLP:** Scikit-learn (TF-IDF)
- **Backend:** Python

---

## Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
**2. Install Dependencies**
```bash
pip install -r requirements.txt
```
**3. Setup Credentials**

**a. Google Gemini**

Add your Gemini API key to the line:

**genai.configure(api_key="YOUR_API_KEY")**

**b. Google Vision Credential (if required):**

Set your environment variable:

**export Google_credential="path/to/your/credentials.json"**

**c. Neo4j**

Make sure Neo4j is running at ** bolt://localhost:7687** with username **neo4j** and password **your_password**

Adjust if different:

**graph = Graph("bolt://localhost:7687", auth=("neo4j", "your_password"))**


---

**Running the App**
```bash
streamlit run your_script.py
```
> Replace your_script.py with your actual file name.

