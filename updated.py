# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import faiss
import re
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import mysql.connector
from mysql.connector import Error

# Optional LLM SDKs
try:
    import openai
except Exception:
    openai = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

# -----------------------------
# ----- CONFIG / CONSTANTS -----
# -----------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP = 50
INDEX_DIR = "rag_index"
FAISS_INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")
VECTORS_FILE = os.path.join(INDEX_DIR, "vectors.npy")

os.makedirs(INDEX_DIR, exist_ok=True)

# -----------------------------
# ----- MySQL Functions -------
# -----------------------------
def create_mysql_connection(host, port, database, user, password):
    """Create and return MySQL connection."""
    try:
        connection = mysql.connector.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        if connection.is_connected():
            return connection
    except Error as e:
        st.error(f"Error connecting to MySQL: {e}")
        return None

def get_table_schema(connection, table_name):
    """Get table schema as a formatted string."""
    try:
        cursor = connection.cursor()
        cursor.execute(f"DESCRIBE {table_name}")
        schema = cursor.fetchall()
        cursor.close()
        
        schema_text = f"Table: {table_name}\nColumns:\n"
        for col in schema:
            schema_text += f"  - {col[0]} ({col[1]})\n"
        return schema_text
    except Error as e:
        st.error(f"Error getting schema: {e}")
        return None

def get_sample_data(connection, table_name, limit=5):
    """Get sample rows from table."""
    try:
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        df = pd.read_sql(query, connection)
        return df
    except Error as e:
        st.error(f"Error getting sample data: {e}")
        return None

def load_table_to_dataframe(connection, table_name):
    """Load entire table into pandas DataFrame."""
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, connection)
        return df
    except Error as e:
        st.error(f"Error loading table: {e}")
        return None

def execute_sql_query(connection, query):
    """Execute SQL query and return results as DataFrame."""
    try:
        df = pd.read_sql(query, connection)
        return df
    except Error as e:
        st.error(f"Error executing query: {e}")
        return None

# -----------------------------
# ----- Helper functions ------
# -----------------------------
@st.cache_resource
def load_embedding_model():
    """Load the sentence-transformers model (cached)."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def simple_clean(text: str) -> str:
    """Light cleaning for text chunks."""
    if text is None:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Chunk by words (simple, deterministic). Returns list of chunk strings."""
    text = simple_clean(text)
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def build_documents_from_df(df: pd.DataFrame, company_col: str = None) -> List[Dict]:
    """Create a list of documents with metadata from a DataFrame."""
    docs = []
    for idx, row in df.iterrows():
        if company_col and company_col in df.columns:
            company = str(row[company_col])
        else:
            company = row.get("company", "unknown") if isinstance(row, dict) else "unknown"

        text_parts = []
        for col in df.columns:
            if col == company_col:
                continue
            text_parts.append(f"{col}: {row[col]}")
        combined = " | ".join([str(x) for x in text_parts])
        combined = f"COMPANY: {company} || ROW:{idx} || " + combined
        
        chunks = chunk_text(combined)
        for j, chunk in enumerate(chunks):
            docs.append({
                "text": chunk,
                "metadata": {"company": company, "row_index": int(idx), "chunk_index": j}
            })
    return docs

def build_faiss_index(docs: List[Dict], embedding_model) -> Tuple[faiss.IndexFlatL2, np.ndarray, List[Dict]]:
    """Build FAISS index from docs."""
    texts = [d["text"] for d in docs]
    metadata = [d["metadata"] for d in docs]

    vectors = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, vectors, metadata

def save_index(index: faiss.IndexFlatL2, vectors: np.ndarray, metadata: List[Dict]):
    faiss.write_index(index, FAISS_INDEX_FILE)
    np.save(VECTORS_FILE, vectors)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

def load_index():
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(METADATA_FILE):
        return None, None, None
    index = faiss.read_index(FAISS_INDEX_FILE)
    vectors = np.load(VECTORS_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
    return index, vectors, metadata

def retrieve_topk(index: faiss.IndexFlatL2, embedding_model, query: str, metadata: List[Dict], k: int = 3) -> List[Dict]:
    """Given a query, returns top-k items."""
    q_vec = embedding_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_vec, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append({
            "score": float(score),
            "text": None,
            "metadata": metadata[idx],
            "vector_index": int(idx)
        })
    return results

# -----------------------------
# ----- LLM FUNCTIONS ---------
# -----------------------------
def select_provider(provider_name: str, api_key: str = None):
    """Central function to switch provider."""
    provider_name = provider_name.lower()
    if provider_name == "openai":
        if openai is None:
            st.warning("openai package not installed.")
        if api_key:
            try:
                openai.api_key = api_key
            except Exception:
                pass
        return {"provider": "openai", "has_sdk": openai is not None}
    
    elif provider_name == "gemini":
        if genai is None:
            st.warning("google-generativeai package not installed.")
        if api_key and genai is not None:
            try:
                genai.configure(api_key=api_key)
            except Exception:
                pass
        return {"provider": "gemini", "has_sdk": genai is not None}
    else:
        return {"provider": "none", "has_sdk": False}

SYSTEM_PROMPT_RAG = (
    "You are an assistant that answers questions ONLY using the user's provided CONTEXT sections. "
    "If the answer is not present in the context, say 'I don't know' or ask for more data. Do not hallucinate."
)

def create_text_to_sql_prompt(question: str, schema: str, sample_data: str = None):
    """Create prompt for text-to-SQL conversion."""
    prompt = f"""Convert the following natural language question into a SQL query.

DATABASE SCHEMA:
{schema}

"""
    if sample_data:
        prompt += f"""SAMPLE DATA:
{sample_data}

"""
    
    prompt += f"""QUESTION: {question}

Generate ONLY the SQL query without any explanation. The query should be executable on MySQL.
Return only the SELECT statement, nothing else."""
    
    return prompt

def call_openai_chat(system_prompt: str, user_prompt: str, api_key: str = None, model: str = "gpt-3.5-turbo"):
    if openai is None:
        return "OpenAI SDK not installed."
    if api_key:
        openai.api_key = api_key
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_prompt}
            ],
            temperature=0.0,
            max_tokens=800,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"OpenAI call error: {e}"

def call_gemini_chat(system_prompt: str, user_prompt: str, api_key: str = None, model: str = "gemini-2.0-flash-exp"):
    if genai is None:
        return "Gemini SDK not installed."
    if api_key:
        try:
            genai.configure(api_key=api_key)
        except Exception:
            pass
    try:
        model_instance = genai.GenerativeModel(model)
        prompt_text = f"{system_prompt}\n\n{user_prompt}"
        response = model_instance.generate_content(prompt_text)
        if hasattr(response, "text"):
            return response.text.strip()
        elif "candidates" in response and len(response["candidates"]) > 0:
            return response["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return str(response)
    except Exception as e:
        return f"Gemini call error: {e}"

def generate_sql_query(question: str, schema: str, sample_data: str, provider_info: dict, api_key: str):
    """Generate SQL query from natural language question."""
    system_prompt = "You are a SQL expert. Convert natural language questions to SQL queries."
    user_prompt = create_text_to_sql_prompt(question, schema, sample_data)
    
    if provider_info["provider"] == "openai" and api_key:
        return call_openai_chat(system_prompt, user_prompt, api_key=api_key)
    elif provider_info["provider"] == "gemini" and api_key:
        return call_gemini_chat(system_prompt, user_prompt, api_key=api_key)
    else:
        return None

def clean_sql_query(sql_text: str) -> str:
    """Clean SQL query from LLM response."""
    # Remove markdown code blocks
    sql_text = re.sub(r'```sql\s*', '', sql_text)
    sql_text = re.sub(r'```\s*', '', sql_text)
    sql_text = sql_text.strip()
    # Remove any trailing semicolon for consistency
    sql_text = sql_text.rstrip(';')
    return sql_text

# -----------------------------
# ----- Streamlit UI ----------
# -----------------------------
st.set_page_config(page_title="MySQL RAG with Text-to-SQL", layout="wide")
st.title("RAG with MySQL + Text-to-SQL Query Generation")

# Sidebar: MySQL Connection
st.sidebar.header("MySQL Connection")
mysql_host = st.sidebar.text_input("Host", value="localhost")
mysql_port = st.sidebar.number_input("Port", value=3306, min_value=1, max_value=65535)
mysql_database = st.sidebar.text_input("Database", value="")
mysql_user = st.sidebar.text_input("User", value="root")
mysql_password = st.sidebar.text_input("Password", type="password")
mysql_table = st.sidebar.text_input("Table Name", value="")

connect_btn = st.sidebar.button("Connect to MySQL")

st.sidebar.markdown("---")
st.sidebar.header("LLM Provider & Keys")
provider_choice = st.sidebar.radio("Choose LLM provider", ("None (local fallback)", "OpenAI", "Gemini"))
api_key_input = st.sidebar.text_input("Paste API key (optional)", type="password")
provider_info = select_provider(provider_choice, api_key_input)

# Session state
if "mysql_connection" not in st.session_state:
    st.session_state["mysql_connection"] = None
if "table_schema" not in st.session_state:
    st.session_state["table_schema"] = None
if "sample_data" not in st.session_state:
    st.session_state["sample_data"] = None
if "docs" not in st.session_state:
    st.session_state["docs"] = None
if "faiss_index" not in st.session_state:
    st.session_state["faiss_index"] = None
if "vectors" not in st.session_state:
    st.session_state["vectors"] = None
if "metadata" not in st.session_state:
    st.session_state["metadata"] = None
if "raw_docs_for_text" not in st.session_state:
    st.session_state["raw_docs_for_text"] = None

embedding_model = load_embedding_model()

# Connect to MySQL
if connect_btn:
    if not mysql_database or not mysql_table:
        st.error("Please provide database name and table name.")
    else:
        connection = create_mysql_connection(mysql_host, mysql_port, mysql_database, mysql_user, mysql_password)
        if connection:
            st.session_state["mysql_connection"] = connection
            st.success("Connected to MySQL successfully!")
            
            # Get schema and sample data
            schema = get_table_schema(connection, mysql_table)
            sample_df = get_sample_data(connection, mysql_table)
            
            if schema:
                st.session_state["table_schema"] = schema
                st.info("Table schema loaded.")
            if sample_df is not None:
                st.session_state["sample_data"] = sample_df.head(5).to_string()
                st.info("Sample data loaded.")

# Display connection status
if st.session_state["mysql_connection"]:
    st.success("✓ MySQL Connected")
    
    # Show schema
    with st.expander("View Table Schema"):
        if st.session_state["table_schema"]:
            st.code(st.session_state["table_schema"])
    
    # Show sample data
    with st.expander("View Sample Data"):
        if st.session_state["sample_data"]:
            st.text(st.session_state["sample_data"])

# Build index from MySQL
st.markdown("---")
st.subheader("Step 1 — Build RAG Index from MySQL Table")

company_col_text = st.text_input("Company column name (if your table has one)", value="company")
build_index_btn = st.button("Build / Rebuild RAG index from MySQL")

if build_index_btn:
    if st.session_state["mysql_connection"] is None:
        st.error("Please connect to MySQL first.")
    else:
        try:
            connection = st.session_state["mysql_connection"]
            df = load_table_to_dataframe(connection, mysql_table)
            
            if df is not None and len(df) > 0:
                docs = build_documents_from_df(df, company_col=company_col_text if company_col_text.strip() else None)
                
                if len(docs) == 0:
                    st.error("No documents extracted from the table.")
                else:
                    st.info(f"Extracted {len(docs)} chunks from MySQL table. Building FAISS index...")
                    index, vectors, metadata = build_faiss_index(docs, embedding_model)
                    save_index(index, vectors, metadata)
                    
                    st.session_state["docs"] = docs
                    st.session_state["faiss_index"] = index
                    st.session_state["vectors"] = vectors
                    st.session_state["metadata"] = metadata
                    st.session_state["raw_docs_for_text"] = docs
                    st.success("RAG index built and saved successfully!")
            else:
                st.error("Could not load data from table.")
        except Exception as e:
            st.error(f"Error building index: {e}")

# Query interface
st.markdown("---")
st.subheader("Step 2 — Ask Questions (Text-to-SQL + RAG)")

query_mode = st.radio("Query Mode", ("Text-to-SQL (Direct Database Query)", "RAG (Semantic Search)"))

query = st.text_input("Enter your question about the data", value="")
ask_btn = st.button("Ask")

if ask_btn and query.strip() != "":
    if st.session_state["mysql_connection"] is None:
        st.error("Please connect to MySQL first.")
    elif query_mode == "Text-to-SQL (Direct Database Query)":
        # Text-to-SQL mode
        if not api_key_input or provider_info["provider"] == "none":
            st.warning("Text-to-SQL requires an LLM provider. Please select OpenAI or Gemini and provide an API key.")
        else:
            with st.spinner("Converting question to SQL..."):
                sql_query = generate_sql_query(
                    query, 
                    st.session_state["table_schema"], 
                    st.session_state["sample_data"],
                    provider_info,
                    api_key_input
                )
                
                if sql_query:
                    sql_query = clean_sql_query(sql_query)
                    st.write("### Generated SQL Query:")
                    st.code(sql_query, language="sql")
                    
                    # Execute query
                    with st.spinner("Executing SQL query..."):
                        result_df = execute_sql_query(st.session_state["mysql_connection"], sql_query)
                        
                        if result_df is not None:
                            st.write("### Query Results:")
                            st.dataframe(result_df)
                            st.write(f"**Rows returned:** {len(result_df)}")
                        else:
                            st.error("Query execution failed. Check the generated SQL.")
                else:
                    st.error("Could not generate SQL query.")
    
    else:
        # RAG mode
        if st.session_state["faiss_index"] is None or st.session_state.get("raw_docs_for_text") is None:
            st.warning("No RAG index available. Build the index first (Step 1).")
        else:
            top_k = st.number_input("Top K retrieved chunks", min_value=1, max_value=10, value=3)
            
            index = st.session_state["faiss_index"]
            metadata = st.session_state["metadata"]
            docs = st.session_state["raw_docs_for_text"]

            # Retrieve
            hits = retrieve_topk(index, embedding_model, query, metadata, k=top_k)
            context_texts = []
            for h in hits:
                idx = h["vector_index"]
                chunk_text = docs[idx]["text"]
                md = docs[idx]["metadata"]
                context_texts.append(f"---\nSource (company={md.get('company')}, row={md.get('row_index')}, chunk={md.get('chunk_index')}):\n{chunk_text}\n")
            
            assembled_context = "\n".join(context_texts)
            user_prompt = f"""Answer the question using ONLY the context sections below.

QUESTION: {query}

CONTEXT:
{assembled_context}

Provide a concise answer and cite sources if helpful."""
            
            st.write("### Retrieved context:")
            st.code(assembled_context[:4000] + ("...\n[truncated]" if len(assembled_context) > 4000 else ""))

            # Call LLM
            if provider_info["provider"] == "openai" and api_key_input:
                with st.spinner("Calling OpenAI..."):
                    res = call_openai_chat(SYSTEM_PROMPT_RAG, user_prompt, api_key=api_key_input)
                    st.markdown("### Answer (OpenAI):")
                    st.write(res)
            elif provider_info["provider"] == "gemini" and api_key_input:
                with st.spinner("Calling Gemini..."):
                    res = call_gemini_chat(SYSTEM_PROMPT_RAG, user_prompt, api_key=api_key_input)
                    st.markdown("### Answer (Gemini):")
                    st.write(res)
            else:
                st.info("No LLM selected — showing retrieved context.")
                st.markdown("**Extracted context:**")
                st.write("\n".join(context_texts))

st.markdown("---")
st.caption("RAG with MySQL + Text-to-SQL powered by Sentence Transformers & FAISS")