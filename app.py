"""
RAG System Frontend - Streamlit Web UI
Interactive interface for the Basecamp Handbook Q&A RAG system
"""

import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Basecamp RAG Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    .header {
        text-align: center;
        padding: 20px;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 30px;
    }
    .answer-box {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 15px 0;
    }
    .source-box {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 3px;
        margin: 10px 0;
        border-left: 4px solid #28a745;
    }
    .chunk-text {
        font-size: 0.9em;
        line-height: 1.5;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="header">
    <h1>📚 Basecamp Handbook Q&A</h1>
    <p><em>Powered by RAG (Retrieval-Augmented Generation)</em></p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - INFORMATION & CONTROLS
# ============================================================================
with st.sidebar:
    st.header("ℹ️ About This System")

    st.markdown("""
    ### What is RAG?
    **Retrieval-Augmented Generation** combines:
    - 🔍 **Semantic Search**: Find relevant handbook sections
    - 🤖 **LLM Generation**: Answer based on actual documents

    ### How It Works
    1. **Loads** Basecamp handbook pages
    2. **Chunks** text into 500-char pieces
    3. **Embeds** using OpenAI's ada-002 model
    4. **Retrieves** top 3 most relevant chunks
    5. **Generates** answers with gpt-3.5-turbo
    """)

    st.divider()

    st.header("🔧 Configuration")
    api_key_status = "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Not Set"
    st.write(f"**OpenAI API Key**: {api_key_status}")

    st.divider()

    st.header("📖 Example Questions")
    example_questions = [
        "What benefits does Basecamp offer employees?",
        "How does Basecamp support work-life balance?",
        "What is Basecamp's approach to internal communication?"
    ]
    for i, q in enumerate(example_questions, 1):
        st.caption(f"{i}. {q}")

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# Initialize session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
    st.session_state.vector_store = None
    st.session_state.retriever = None
    st.session_state.loading_complete = False

# ============================================================================
# INITIALIZATION - LOAD AND PREPARE RAG SYSTEM
# ============================================================================

if not st.session_state.loading_complete:
    with st.spinner("🚀 Initializing RAG System..."):
        try:
            progress_bar = st.progress(0)

            # Check API key
            if not os.getenv("OPENAI_API_KEY"):
                st.error("❌ **Error**: OPENAI_API_KEY environment variable is not set!")
                st.info("Please set your OpenAI API key and restart the app.")
                st.stop()

            # Step 1: Load Documents
            progress_bar.progress(15)
            st.write("📥 **Step 1**: Loading Basecamp Handbook...")

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

            loader = WebBaseLoader(urls)
            documents = loader.load()
            st.success(f"✓ Loaded {len(documents)} documents")

            # Step 2: Split Documents
            progress_bar.progress(30)
            st.write("✂️ **Step 2**: Splitting Documents...")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
            chunks = text_splitter.split_documents(documents)
            st.success(f"✓ Created {len(chunks)} chunks")

            # Step 3: Create Embeddings & Vector Store
            progress_bar.progress(50)
            st.write("🔢 **Step 3**: Creating Embeddings (this may take a moment)...")

            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            st.session_state.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name="basecamp-handbook"
            )
            st.success(f"✓ Embedded {len(chunks)} chunks in ChromaDB")

            # Step 4: Setup Retriever
            progress_bar.progress(75)
            st.write("🔍 **Step 4**: Setting Up Retriever...")

            st.session_state.retriever = st.session_state.vector_store.as_retriever(
                search_kwargs={"k": 3}
            )
            st.success("✓ Retriever ready (returns top 3 chunks)")

            # Step 5: Setup LLM & Chain
            progress_bar.progress(90)
            st.write("🤖 **Step 5**: Initializing Language Model...")

            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

            prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions about Basecamp's employee handbook.
Use the following context from the handbook to answer the question.
If the answer is not in the context, say "I don't have information about that in the handbook."

Context from the handbook:
{context}

Question: {question}

Answer:
""")

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            st.session_state.rag_chain = (
                {
                    "context": st.session_state.retriever | (lambda docs: format_docs(docs)),
                    "question": RunnablePassthrough(),
                }
                | prompt_template
                | llm
                | StrOutputParser()
            )
            st.success("✓ RAG chain initialized!")

            progress_bar.progress(100)
            st.session_state.loading_complete = True
            st.success("✅ **RAG System Ready!** You can now ask questions.")

        except Exception as e:
            st.error(f"❌ **Error during initialization**: {str(e)}")
            st.info("Please check your OpenAI API key and try again.")
            st.stop()

# ============================================================================
# QUERY INTERFACE
# ============================================================================

st.markdown("---")
st.header("❓ Ask a Question")

# Create columns for input
col1, col2 = st.columns([4, 1])

with col1:
    user_query = st.text_input(
        "Enter your question about Basecamp:",
        placeholder="e.g., What are Basecamp's core values?",
        key="query_input"
    )

with col2:
    search_button = st.button("🔍 Search", use_container_width=True)

# ============================================================================
# PROCESS QUERY AND DISPLAY RESULTS
# ============================================================================

if search_button and user_query:
    with st.spinner("🤖 Searching and generating answer..."):
        try:
            # Get the answer
            answer = st.session_state.rag_chain.invoke(user_query)

            # Get the source chunks
            retrieved_docs = st.session_state.retriever.invoke(user_query)

            # Display Answer
            st.markdown("### ✅ Answer")
            st.markdown(f"""
<div class="answer-box">
    {answer}
</div>
""", unsafe_allow_html=True)

            # Display Source Chunks
            st.markdown("### 📚 Source Chunks (Top 3 Retrieved)")

            for i, doc in enumerate(retrieved_docs, 1):
                with st.expander(f"📄 Chunk {i}", expanded=(i == 1)):
                    st.markdown(f"""
<div class="source-box">
    <div class="chunk-text">
        {doc.page_content}
    </div>
</div>
""", unsafe_allow_html=True)

            st.success("✅ Done!")

        except Exception as e:
            st.error(f"❌ **Error generating answer**: {str(e)}")
            if "insufficient_quota" in str(e):
                st.warning("⚠️ Your OpenAI API key has insufficient quota. Please check your billing.")
            elif "401" in str(e) or "invalid" in str(e):
                st.warning("⚠️ Your OpenAI API key is invalid. Please check your key.")

# ============================================================================
# QUICK EXAMPLE QUERIES
# ============================================================================

st.markdown("---")
st.header("⚡ Quick Examples")

col1, col2, col3 = st.columns(3)

example_queries = [
    "What benefits does Basecamp offer employees?",
    "How does Basecamp support work-life balance?",
    "What is Basecamp's approach to internal communication?"
]

with col1:
    if st.button(example_queries[0], use_container_width=True):
        st.session_state.query_input = example_queries[0]
        st.rerun()

with col2:
    if st.button(example_queries[1], use_container_width=True):
        st.session_state.query_input = example_queries[1]
        st.rerun()

with col3:
    if st.button(example_queries[2], use_container_width=True):
        st.session_state.query_input = example_queries[2]
        st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 20px;">
    <p>🤖 Powered by LangChain, ChromaDB, and OpenAI</p>
    <p>Tutorial: <a href="https://www.youtube.com/watch?v=sVcwVQRHIc8">Learn RAG From Scratch</a></p>
</div>
""", unsafe_allow_html=True)
