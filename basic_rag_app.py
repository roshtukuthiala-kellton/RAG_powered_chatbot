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

# Optional LLM SDKs (only used if you provide API keys)
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
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # free, small, accurate
CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP = 50
INDEX_DIR = "rag_index"
FAISS_INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")
VECTORS_FILE = os.path.join(INDEX_DIR, "vectors.npy")

# ensures index dir exists
os.makedirs(INDEX_DIR, exist_ok=True)

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
    """
    Create a list of documents with metadata from a DataFrame.
    Each returned dict contains:
      - 'text' : chunk string
      - 'metadata' : dict with company, row index, original row as dict
    """
    docs = []
    for idx, row in df.iterrows():
        # combine row into single text
        if company_col and company_col in df.columns:
            company = str(row[company_col])
        else:
            # if no company col, leave blank or set to 'unknown'
            company = row.get("company", "unknown") if isinstance(row, dict) else "unknown"

        # combine all columns except maybe the company column
        text_parts = []
        for col in df.columns:
            if col == company_col:
                continue
            text_parts.append(f"{col}: {row[col]}")
        combined = " | ".join([str(x) for x in text_parts])
        combined = f"COMPANY: {company} || ROW:{idx} || " + combined
        # chunk
        chunks = chunk_text(combined)
        for j, chunk in enumerate(chunks):
            docs.append({
                "text": chunk,
                "metadata": {"company": company, "row_index": int(idx), "chunk_index": j}
            })
    return docs

def build_faiss_index(docs: List[Dict], embedding_model) -> Tuple[faiss.IndexFlatL2, np.ndarray, List[Dict]]:
    """
    Build FAISS index from docs (list of {'text','metadata'}).
    Returns (index, vectors_np_array, metadata_list)
    """
    texts = [d["text"] for d in docs]
    metadata = [d["metadata"] for d in docs]

    # embed
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
    """
    Given a query, returns top-k items as list of dicts:
      {'score': float, 'text': str, 'metadata': dict}
    """
    q_vec = embedding_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_vec, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append({
            "score": float(score),
            "text": None,   # we don't keep all texts here - see note below
            "metadata": metadata[idx],
            "vector_index": int(idx)
        })
    return results

def get_text_by_index(docs: List[Dict], idx: int) -> str:
    """Return text for a doc index (helper when we persisted only metadata)."""
    return docs[idx]["text"]

# -----------------------------
# ----- LLM SWITCH FUNCTION ---
# -----------------------------
def select_provider(provider_name: str, api_key: str = None):
    """
    central function to 'switch' provider. It sets up the SDK if available.
    - provider_name: "OpenAI" | "Gemini" | "None"
    - api_key: optional API key string (if you want to set it here)
    Returns a dict with 'provider' and any info needed.
    """
    provider_name = provider_name.lower()
    if provider_name == "openai":
        if openai is None:
            st.warning("openai package not installed. Install `openai` to use OpenAI provider.")
        if api_key:
            try:
                openai.api_key = api_key
            except Exception:
                pass
        return {"provider": "openai", "has_sdk": openai is not None}
    
    elif provider_name == "gemini":
        if genai is None:
            st.warning("google-generativeai package not installed. Install `google-generativeai` to use Gemini provider.")
        if api_key and genai is not None:
            try:
                genai.configure(api_key=api_key)
            except Exception:
                pass
        return {"provider": "gemini", "has_sdk": genai is not None}
    else:
        return {"provider": "none", "has_sdk": False}

# -----------------------------
# ----- LLM CALL WRAPPERS -----
# -----------------------------
SYSTEM_PROMPT = (
    "You are an assistant that answers questions ONLY using the user's provided CONTEXT sections. "
    "If the answer is not present in the context, say 'I don't know' or ask for more data. Do not hallucinate."
)

def call_openai_chat(system_prompt: str, user_prompt: str, api_key: str = None, model: str = "gpt-3.5-turbo"):
    if openai is None:
        return "OpenAI SDK not installed. Can't call OpenAI."
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

# def call_gemini_chat(system_prompt: str, user_prompt: str, api_key: str = None, model: str = "gpt-4o-mini"):
#     if genai is None:
#         return "Gemini SDK (google-generativeai) not installed. Can't call Gemini."
#     if api_key:
#         try:
#             genai.configure(api_key=api_key)
#         except Exception:
#             pass
#     try:
#         # Note: google generative ai client API surfaces can change with versions.
#         # This snippet follows the general pattern but you might need to adjust to your installed package.
#         resp = genai.chat.completions.create(
#             model=model,
#             messages=[{"role":"system","content":system_prompt}, {"role":"user","content":user_prompt}],
#             temperature=0.0,
#             max_output_tokens=800
#         )
#         # There are variant response shapes; try to safely extract text:
#         if hasattr(resp, "candidates"):
#             return resp.candidates[0].content
#         elif "candidates" in resp and len(resp["candidates"])>0:
#             return resp["candidates"][0]["content"]
#         elif "outputs" in resp and len(resp["outputs"])>0:
#             return resp["outputs"][0]["content"]
#         else:
#             return str(resp)
#     except Exception as e:
#         return f"Gemini call error: {e}"

def call_gemini_chat(system_prompt: str, user_prompt: str, api_key: str = None, model: str = "gemini-2.5-flash"):
    if genai is None:
        return "Gemini SDK (google-generativeai) not installed. Can't call Gemini."
    if api_key:
        try:
            genai.configure(api_key=api_key)
        except Exception:
            pass
    try:
        # create a model instance
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


# -----------------------------
# ----- Streamlit UI ----------
# -----------------------------
st.set_page_config(page_title="Excel → RAG (Streamlit)", layout="wide")
st.title("RAG with Excel + Free Embeddings (toggle: OpenAI / Gemini / None)")

# Sidebar: provider toggle + keys
st.sidebar.header("LLM Provider & Keys")
provider_choice = st.sidebar.radio("Choose LLM provider", ("None (local fallback)", "OpenAI", "Gemini"))
api_key_input = st.sidebar.text_input("Paste API key (optional)", type="password")
# set provider using central function
provider_info = select_provider(provider_choice, api_key_input)

st.sidebar.markdown("---")
st.sidebar.markdown("Free embeddings: **sentence-transformers** (no paid embeddings required).")

# Main UI: upload
st.subheader("Step 1 — Upload your Excel file (one sheet or several)")

uploaded_file = st.file_uploader("Upload .xlsx or .csv", type=["xlsx","xls","csv"])
company_col_text = st.text_input("Company column name (if your sheet has one). Leave blank if companies are in separate sheets.", value="company")

build_index_btn = st.button("Build / Rebuild index from uploaded file")

# persistent in session
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

if uploaded_file is not None and build_index_btn:
    # Load file - either csv or excel with sheets
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            docs = build_documents_from_df(df, company_col=company_col_text if company_col_text.strip() else None)
        else:
            # read all sheets and concatenate
            xls = pd.read_excel(uploaded_file, sheet_name=None)
            all_docs = []
            for sheet_name, df in xls.items():
                # if file has multiple sheets and no company col, mark company as sheet name
                if company_col_text.strip() == "" or company_col_text not in df.columns:
                    # add 'company' column with sheet name (if missing)
                    if "company" not in df.columns:
                        df["company"] = sheet_name
                docs_sheet = build_documents_from_df(df, company_col=company_col_text if company_col_text.strip() else "company")
                all_docs.extend(docs_sheet)
            docs = all_docs

        if len(docs) == 0:
            st.error("No documents extracted from the file. Check file format.")
        else:
            st.info(f"Extracted {len(docs)} chunks from the uploaded file. Building FAISS index (this may take a moment)...")
            index, vectors, metadata = build_faiss_index(docs, embedding_model)
            # Save to disk and to session
            save_index(index, vectors, metadata)
            st.session_state["docs"] = docs
            st.session_state["faiss_index"] = index
            st.session_state["vectors"] = vectors
            st.session_state["metadata"] = metadata
            st.session_state["raw_docs_for_text"] = docs  # store texts so we can show sources
            st.success("Index built and saved.")
    except Exception as e:
        st.error(f"Error building index: {e}")

# Try to load index if exists and session empty
if st.session_state["faiss_index"] is None:
    loaded = load_index()
    if loaded[0] is not None:
        st.session_state["faiss_index"], st.session_state["vectors"], st.session_state["metadata"] = loaded
        # we cannot reconstruct docs texts reliably from saved vectors+metadata alone;
        # if we saved full docs in a session earlier it is available. If not, user should rebuild.
        if os.path.exists(os.path.join(INDEX_DIR, "docs.pkl")):
            with open(os.path.join(INDEX_DIR, "docs.pkl"), "rb") as f:
                st.session_state["raw_docs_for_text"] = pickle.load(f)
        st.info("Loaded existing FAISS index from disk.")

st.markdown("---")
st.subheader("Step 2 — Ask questions (RAG)")

if st.session_state["faiss_index"] is None or st.session_state.get("raw_docs_for_text") is None:
    st.warning("No index available. Upload and build the index first (Step 1).")
else:
    query = st.text_input("Enter your question about the data", value="")
    top_k = st.number_input("Top K retrieved chunks", min_value=1, max_value=10, value=3)
    ask_btn = st.button("Ask")

    if ask_btn and query.strip() != "":
        index = st.session_state["faiss_index"]
        metadata = st.session_state["metadata"]
        docs = st.session_state["raw_docs_for_text"]

        # Retrieve
        hits = retrieve_topk(index, embedding_model, query, metadata, k=top_k)
        # Build context text from hits
        context_texts = []
        for h in hits:
            idx = h["vector_index"]
            # retrieve the actual text chunk from docs stored in session
            chunk_text = docs[idx]["text"]
            md = docs[idx]["metadata"]
            context_texts.append(f"---\nSource (company={md.get('company')}, row={md.get('row_index')}, chunk={md.get('chunk_index')}):\n{chunk_text}\n")
        assembled_context = "\n".join(context_texts)

        # Compose prompt for LLM
        user_prompt = f"""Answer the question using ONLY the context sections below. If the answer is not contained in the context, say "I don't know" or ask for more data.\n\nQUESTION: {query}\n\nCONTEXT:\n{assembled_context}\n\nProvide a concise answer and, if helpful, cite which sources (company/row) you used."""
        st.write("### Retrieved context (shown to the model):")
        st.code(assembled_context[:4000] + ("...\n[truncated]" if len(assembled_context) > 4000 else ""))

        # Call chosen provider or fallback
        if provider_info["provider"] == "openai":
            if api_key_input.strip() == "":
                st.warning("No OpenAI API key provided in the sidebar — can't call OpenAI. Showing context as fallback.")
                st.markdown("**Fallback:** returning retrieved context excerpts since no key provided.")
                st.write("\n".join(context_texts))
            else:
                with st.spinner("Calling OpenAI..."):
                    res = call_openai_chat(SYSTEM_PROMPT, user_prompt, api_key=api_key_input, model="gpt-3.5-turbo")
                    st.markdown("### Answer (OpenAI):")
                    st.write(res)
        elif provider_info["provider"] == "gemini":
            if api_key_input.strip() == "":
                st.warning("No Gemini API key provided in the sidebar — can't call Gemini. Showing context as fallback.")
                st.markdown("**Fallback:** returning retrieved context excerpts since no key provided.")
                st.write("\n".join(context_texts))
            else:
                with st.spinner("Calling Gemini..."):
                    res = call_gemini_chat(SYSTEM_PROMPT, user_prompt, api_key=api_key_input, model="gemini-2.5-flash")
                    st.markdown("### Answer (Gemini):")
                    st.write(res)
        else:
            # No LLM — local fallback: just show the retrieved context and a small naive summary
            st.info("No LLM selected — performing a local (non-LLM) fallback: showing retrieved context and a short automated summary.")
            st.markdown("**Extracted context:**")
            st.write("\n".join(context_texts))
            # Naive local summary: return first 1-2 sentences from each chunk
            def naive_summary(texts):
                sents = []
                for t in texts:
                    t = t.strip()
                    # take up to first 200 characters as a naive 'summary'
                    sents.append(t[:200] + ("..." if len(t) > 200 else ""))
                return "\n".join(sents)
            st.markdown("**Naive local summary (no LLM):**")
            st.write(naive_summary(context_texts))

st.markdown("---")
