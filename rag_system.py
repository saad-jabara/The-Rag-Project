"""
RAG System Implementation - Learn RAG From Scratch
Tutorial: https://www.youtube.com/watch?v=sVcwVQRHIc8

This script implements a complete Retrieval-Augmented Generation system that:
1. Loads documents from the Basecamp Employee Handbook
2. Chunks them into manageable pieces
3. Creates embeddings and stores them in ChromaDB
4. Retrieves relevant chunks for user queries
5. Generates answers using OpenAI's GPT-3.5-turbo

Prerequisites:
- OPENAI_API_KEY environment variable must be set
"""

# ============================================================================
# STEP 0: IMPORTS & CONFIGURATION
# ============================================================================
print("=" * 70)
print("RAG System - Basecamp Handbook Question Answering")
print("=" * 70)
print("\n[STEP 0] Importing libraries...\n")

import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Check if OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ Error: OPENAI_API_KEY environment variable is not set!")

print("✓ All libraries imported successfully\n")

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================
print("=" * 70)
print("[STEP 1] DATA LOADING - Loading Basecamp Handbook Pages")
print("=" * 70)
print("\nLoading ~10 pages from Basecamp Employee Handbook...\n")

# URLs of the Basecamp handbook pages
urls = [
    "https://basecamp.com/handbook",
    "https://basecamp.com/handbook/how-we-work",
    "https://basecamp.com/handbook/benefits-and-perks",
    "https://basecamp.com/handbook/work-life-balance",
    "https://basecamp.com/handbook/titles-for-support",
    "https://basecamp.com/handbook/getting-started",
    "https://basecamp.com/handbook/communication",
    "https://basecamp.com/handbook/our-internal-systems",
    "https://basecamp.com/handbook/pricing-and-profit",
    "https://basecamp.com/handbook/dei",
]

# Use WebBaseLoader to load content from the URLs
loader = WebBaseLoader(urls)
documents = loader.load()

# Print statistics about loaded documents
print(f"✓ Successfully loaded {len(documents)} document(s)")
print(f"\nFirst document preview (first 300 characters):")
print("-" * 70)
print(documents[0].page_content[:300] + "...\n")

# ============================================================================
# STEP 2: TEXT SPLITTING
# ============================================================================
print("=" * 70)
print("[STEP 2] TEXT SPLITTING - Chunking Documents")
print("=" * 70)
print("\nSplitting documents into chunks (size=500, overlap=100)...\n")

# Configure the text splitter
# chunk_size: How big each chunk is (in characters)
# chunk_overlap: How much chunks overlap with each other
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# Split all documents into chunks
chunks = text_splitter.split_documents(documents)

# Print statistics
print(f"✓ Documents split into {len(chunks)} chunks")
print(f"\nExample chunk (first 250 characters):")
print("-" * 70)
print(chunks[0].page_content[:250] + "...\n")

# ============================================================================
# STEP 3: EMBEDDINGS & VECTOR STORE
# ============================================================================
print("=" * 70)
print("[STEP 3] EMBEDDINGS & VECTOR STORE - Creating Embeddings with ChromaDB")
print("=" * 70)
print("\nCreating embeddings for all chunks...")
print("(Using OpenAI's text-embedding-ada-002 model - 1536 dimensions)\n")

# Initialize the embeddings model
# text-embedding-ada-002: OpenAI's latest embedding model (1536 dimensions)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Create a vector store using ChromaDB
# This stores all embeddings and allows efficient similarity search
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="basecamp-handbook"
)

print("✓ Embeddings created and indexed in ChromaDB")
print(f"✓ Total vectors in store: {len(chunks)}\n")

# ============================================================================
# STEP 4: RETRIEVAL CONFIGURATION
# ============================================================================
print("=" * 70)
print("[STEP 4] RETRIEVAL - Setting Up Retriever")
print("=" * 70)
print("\nConfiguring retriever to return top 3 most relevant chunks...\n")

# Create a retriever that returns the top 3 most similar chunks
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

print("✓ Retriever configured to return top 3 chunks\n")

# ============================================================================
# STEP 5: GENERATION - LLM CONFIGURATION
# ============================================================================
print("=" * 70)
print("[STEP 5] GENERATION - Setting Up Language Model")
print("=" * 70)
print("\nInitializing ChatOpenAI (gpt-3.5-turbo)...\n")

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Create the prompt template that tells the LLM how to answer
# It will receive:
# - {context}: The retrieved chunks
# - {question}: The user's question
prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions about Basecamp's employee handbook.
Use the following context from the handbook to answer the question.
If the answer is not in the context, say "I don't have information about that in the handbook."

Context from the handbook:
{context}

Question: {question}

Answer:
""")

# Function to format retrieved documents as a single string
def format_docs(docs):
    """Convert retrieved documents into a single formatted string."""
    return "\n\n".join(doc.page_content for doc in docs)

# Create the RAG chain: retriever -> prompt -> LLM -> output
# This chains all components together:
# 1. Retrieve relevant chunks based on the question
# 2. Format them as context
# 3. Pass to the prompt template with the question
# 4. Send to LLM
# 5. Extract the answer text
rag_chain = (
    {
        "context": retriever | (lambda docs: format_docs(docs)),
        "question": RunnablePassthrough(),
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

print("✓ RAG chain created and ready to answer questions!\n")

# ============================================================================
# STEP 6: EXAMPLE QUERIES
# ============================================================================
print("=" * 70)
print("[STEP 6] EXAMPLE QUERIES - Testing the RAG System")
print("=" * 70)

# List of example questions to test
example_queries = [
    "What benefits does Basecamp offer employees?",
    "How does Basecamp support work-life balance?",
    "What is Basecamp's approach to internal communication?",
]

# Process each query
for i, query in enumerate(example_queries, 1):
    print(f"\n{'─' * 70}")
    print(f"Query #{i}")
    print(f"{'─' * 70}")
    print(f"\n❓ Question: {query}\n")

    # Generate the answer
    answer = rag_chain.invoke(query)

    print(f"✓ Answer:\n{answer}\n")

    # Retrieve and show the source chunks
    retrieved_docs = retriever.invoke(query)
    print(f"\n📚 Retrieved Source Chunks (Top 3):")
    print("-" * 70)
    for j, doc in enumerate(retrieved_docs, 1):
        print(f"\nChunk {j}:")
        print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)

print(f"\n{'=' * 70}")
print("✓ RAG System demonstration complete!")
print("=" * 70)
print("\nYou can now use the 'rag_chain' to answer your own questions.")
print("Simply call: rag_chain.invoke('Your question here')\n")

# ============================================================================
# OPTIONAL: INTERACTIVE MODE
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE - Ask Your Own Questions")
    print("=" * 70)
    print("(Type 'exit' to quit)\n")

    while True:
        user_query = input("Ask a question about Basecamp: ").strip()

        if user_query.lower() == "exit":
            print("\nThank you for using the RAG system!")
            break

        if not user_query:
            continue

        print(f"\n🤖 Searching and generating answer...\n")
        answer = rag_chain.invoke(user_query)
        print(f"Answer:\n{answer}\n")
