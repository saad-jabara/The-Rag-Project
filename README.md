# RAG System - Basecamp Handbook Q&A

A complete **Retrieval-Augmented Generation (RAG)** system in Python that answers questions about the Basecamp Employee Handbook using LLMs and vector embeddings.

## What is RAG?

**Retrieval-Augmented Generation (RAG)** combines two powerful AI techniques:

1. **Retrieval**: Using semantic search to find relevant document chunks related to a question
2. **Generation**: Using a language model to generate accurate answers based on the retrieved context

Instead of relying solely on a pre-trained LLM's knowledge (which can be outdated or hallucinate), RAG systems:
- Load specific documents (your knowledge base)
- Convert them to embeddings (semantic vectors)
- Retrieve the most relevant chunks for each query
- Pass them as context to the LLM
- Generate grounded, factual answers

**Benefits:**
- ✓ Answers based on your actual documents
- ✓ Cites sources automatically
- ✓ Reduces hallucinations
- ✓ Keeps answers up-to-date with document changes

## What This Project Does

This project implements a complete RAG pipeline that:

1. **Loads** ~10 pages from the Basecamp Employee Handbook using web scraping
2. **Splits** documents into overlapping chunks (500 chars, 100 char overlap)
3. **Embeds** all chunks using OpenAI's `text-embedding-ada-002` model (1536 dimensions)
4. **Stores** vectors in ChromaDB for efficient similarity search
5. **Retrieves** the top 3 most relevant chunks for each query
6. **Generates** answers using `gpt-3.5-turbo`
7. **Shows** source chunks used to answer each question

## Tech Stack

- **LangChain**: Framework for building LLM applications
- **ChromaDB**: Vector database for storing embeddings
- **OpenAI API**: Embeddings (ada-002) and LLM (gpt-3.5-turbo)
- **Python 3.8+**: Programming language

## Setup & Installation

### 1. Clone the Repository

```bash
git clone git@github.com:saad-jabara/The-Rag-Project.git
cd The-Rag-Project
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Your OpenAI API Key

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

On Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY = "your-openai-api-key-here"
```

### 5. Run the Script

```bash
python rag_system.py
```

## How It Works - Step by Step

### Step 1: Data Loading
The script loads ~10 pages from Basecamp's handbook using `WebBaseLoader`:
- Handbook overview
- How we work
- Benefits and perks
- Work-life balance
- Titles and roles
- Getting started
- Communication guidelines
- Internal systems
- Pricing and profit
- Diversity, equity & inclusion

```python
loader = WebBaseLoader(urls)
documents = loader.load()
```

### Step 2: Text Splitting
Documents are split into manageable chunks:
- **Chunk size**: 500 characters (enough context per chunk)
- **Overlap**: 100 characters (maintains context between chunks)

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)
```

### Step 3: Embeddings & Vector Store
All chunks are converted to embeddings and stored in ChromaDB:
- **Embedding model**: `text-embedding-ada-002` (1536 dimensions)
- **Similarity metric**: Cosine similarity
- **Vector database**: ChromaDB (in-memory, can be persisted)

```python
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = Chroma.from_documents(chunks, embedding=embeddings)
```

### Step 4: Retrieval
For each query, retrieve the top 3 most semantically similar chunks:

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
```

### Step 5: Generation
Pass the query + retrieved context to GPT-3.5-turbo:

```python
llm = ChatOpenAI(model="gpt-3.5-turbo")
rag_chain = ({...} | prompt_template | llm | StrOutputParser())
```

### Step 6: Answer & Sources
The system outputs:
- The generated answer
- The top 3 source chunks used

## Example Usage

### Running Example Queries

The script automatically runs three example queries:

```
Query #1
❓ Question: What benefits does Basecamp offer employees?
✓ Answer: [Generated answer based on retrieved chunks]

Query #2
❓ Question: How does Basecamp support work-life balance?
✓ Answer: [Generated answer based on retrieved chunks]

Query #3
❓ Question: What is Basecamp's approach to internal communication?
✓ Answer: [Generated answer based on retrieved chunks]
```

### Interactive Mode

After the examples run, you can ask your own questions:

```bash
Ask a question about Basecamp: How do I request time off?
🤖 Searching and generating answer...

Answer:
[Answer about time off policies]

Ask a question about Basecamp: exit
```

### Example Answers

**Q: What benefits does Basecamp offer employees?**

A: Based on the Basecamp handbook, the company offers comprehensive benefits including:
- Paid time off
- Health insurance
- Professional development opportunities
- Flexible work arrangements
- And more...

(Note: Actual answers will vary based on handbook content retrieval)

## Project Structure

```
The-Rag-Project/
├── rag_system.py          # Main RAG system script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Key Concepts Explained

### Embeddings
- Convert text into numerical vectors (1536 dimensions for ada-002)
- Similar meaning = similar vectors
- Enables semantic search beyond keyword matching

### Vector Similarity Search
- Find chunks with vectors closest to the query vector
- Returns top K most relevant chunks (K=3 in this project)

### Context Window
- Pass retrieved chunks as context to the LLM
- LLM generates answers grounded in actual documents
- Reduces hallucinations and improves accuracy

## Troubleshooting

### Issue: `OPENAI_API_KEY not set`
**Solution**: Make sure your API key is exported:
```bash
export OPENAI_API_KEY="sk-..."
```

### Issue: `ModuleNotFoundError`
**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: Rate limit errors from OpenAI
**Solution**: Wait a moment and retry. Consider upgrading your OpenAI account for higher limits.

### Issue: ChromaDB not persisting
**Solution**: By default, the vector store is in-memory. To persist between runs, modify the Chroma initialization:
```python
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

## Next Steps & Enhancements

1. **Persist Vector Store**: Save ChromaDB locally to avoid re-embedding on every run
2. **Multiple Document Formats**: Support PDF, Word, Markdown files
3. **Streaming Responses**: Stream LLM responses for faster feedback
4. **Web UI**: Build a Streamlit or FastAPI web interface
5. **Advanced Chunking**: Implement semantic chunking or sliding windows
6. **Hybrid Search**: Combine keyword search with semantic search
7. **Multi-Turn Conversations**: Add memory for context across multiple questions
8. **Cost Optimization**: Use cheaper embedding models or local embeddings

## Tutorial Credit

This project is inspired by and follows the principles from:

**"Learn RAG From Scratch – Python AI Tutorial from a LangChain Engineer"**
- 📺 YouTube: https://www.youtube.com/watch?v=sVcwVQRHIc8
- Instructor: A LangChain engineer
- Full tutorial demonstrating production-ready RAG implementation

## License

This project is educational and follows the same license as the tutorial it's based on.

## Author

Created as a hands-on implementation of RAG concepts using LangChain, ChromaDB, and OpenAI APIs.

---

**Happy RAG building! 🚀**

For questions or improvements, feel free to open an issue or submit a pull request.
