#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:33:45 2026
@author: sigillus
"""

import os
import streamlit as st
import psycopg
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# ====================== CONNECTION ======================
# Supabase Transaction Pooler (Port 6543) - wichtig für Streamlit Cloud
if "DATABASE_URL" in st.secrets:
    db_connection_str = st.secrets["DATABASE_URL"]
else:
    # Fallback für lokale Entwicklung
    load_dotenv()
    db_password = os.getenv("DB_PASSWORD")
    if not db_password:
        st.error("DB_PASSWORD nicht gefunden!")
        st.stop()
    db_connection_str = f"postgresql+psycopg://postgres:{db_password}@db.advtrcewqqpkenlpbjch.supabase.co:6543/postgres?sslmode=require"

raw_conn_string = db_connection_str.replace("postgresql+psycopg://", "postgresql://")
# =======================================================

# --- 2. TEXTE AUS DB LADEN (für BM25) ---
def get_all_texts_from_db(conn_str):
    texts = []
    try:
        with psycopg.connect(conn_str, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name NOT LIKE 'langchain_%'
                    AND table_name NOT LIKE 'pg_%'
                """)
                tables = [row[0] for row in cur.fetchall()]
               
                for table in tables:
                    cur.execute("""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = %s
                        AND data_type IN ('text', 'varchar', 'character varying')
                    """, (table,))
                    text_columns = [row[0] for row in cur.fetchall()]
                   
                    for col in text_columns:
                        cur.execute(f'SELECT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL')
                        rows = cur.fetchall()
                        texts.extend([row[0] for row in rows if row[0]])
    except Exception as e:
        st.warning(f"Fehler beim Laden der Texte für BM25: {e}")
    return texts

# --- 3. MODELLE & EMBEDDINGS ---
qui = os.getenv("TOGETHER_API_KEY")
if not qui and "TOGETHER_API_KEY" in st.secrets:
    qui = st.secrets["TOGETHER_API_KEY"]

model = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=qui,
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# --- 4. RETRIEVER SETUP ---
@st.cache_resource(show_spinner="Initialisiere Vectorstore...")
def init_vectorstore():
    try:
        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name="GP_knowledge",
            connection=db_connection_str,
            create_extension=False,
            use_jsonb=True,
        )
        return vectorstore
    except Exception as e:
        st.error(f"Fehler beim Initialisieren des Vectorstores: {e}")
        st.stop()

vectorstore = init_vectorstore()
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

# BM25
all_texts = get_all_texts_from_db(raw_conn_string)
bm25_retriever = None
if all_texts:
    bm25_retriever = BM25Retriever.from_texts(all_texts)
    bm25_retriever.k = 12

# SQL Database
db = SQLDatabase.from_uri(db_connection_str)
sql_toolkit = SQLDatabaseToolkit(db=db, llm=model)
sql_tools = sql_toolkit.get_tools()

# --- 5. HILFSFUNKTIONEN FÜR DIE SUCHE ---
def hybrid_retrieval_with_sources(query, k=60):
    query = query.replace("Welche Risiken hat ", "").replace("?", "").strip()
   
    v_docs = vector_retriever.invoke(query)
    b_docs = bm25_retriever.invoke(query) if bm25_retriever else []
   
    doc_scores = {}
   
    for rank, doc in enumerate(v_docs, start=1):
        doc_scores[doc.page_content] = {"score": 1 / (rank + k), "doc": doc}
       
    for rank, doc in enumerate(b_docs, start=1):
        content = doc.page_content
        if content in doc_scores:
            doc_scores[content]["score"] += 1 / (rank + k)
        else:
            doc_scores[content] = {"score": 1 / (rank + k), "doc": doc}
   
    reranked_results = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
   
    context_parts = []
    sources = set()

    # SQL-Suche
    try:
        sql_result = db.run(f"SELECT * FROM priscus WHERE wirkstoff ILIKE '%{query}%' LIMIT 5;")
        if sql_result:
            context_parts.append(f"Priscus-Liste:\n{sql_result}")
    except:
        pass

    try:
        sql_result = db.run(f"SELECT * FROM forta_liste WHERE wirkstoff ILIKE '%{query}%' LIMIT 5;")
        if sql_result:
            context_parts.append(f"FORTA-Liste:\n{sql_result}")
    except:
        pass

    for item in reranked_results:
        doc = item["doc"]
        context_parts.append(doc.page_content)
        source_name = doc.metadata.get("source", "Unbekannte Quelle")
        source_file = os.path.basename(source_name)
        sources.add(f"- {source_file}")

    return {
        "context": "\n\n---\n\n".join(context_parts),
        "sources": "\n".join(sources)
    }

# --- 6. PROMPT SETUP ---
instructions = (
    "Du bist ein spezialisierter medizinischer Experte für Medikamentensicherheit und Praxisführung. "
    "Deine Wissensbasis sind AUSSCHLIESSLICH die bereitgestellten Textstellen. "
    "Antworte NUR auf Basis des unten gelieferten Kontextes.\n\n"
)

prompt = ChatPromptTemplate.from_template(
    instructions + "\n\nKONTEXT:\n{context}\n\nFRAGE: {question}\n\nANTWORT:"
)

# --- 7. HAUPTFUNKTION ---
def run_smart_query_streamlit(question):
    data = hybrid_retrieval_with_sources(question)
   
    if not data["context"]:
        return "⚠️ Keine Treffer in der Datenbank."
   
    current_chain = prompt | model
        
    try:
        result = current_chain.invoke({
            "context": data["context"],
            "question": question
        })
        return result.content
    except Exception as e:
        return f"Fehler bei der Verarbeitung: {e}"

# ====================== STREAMLIT UI ======================
st.set_page_config(
    page_title="ChatBot für praktische Ärzte",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 prototypischer ChatBot mit Sonderwissen für praktische Ärzte")

st.markdown("""
Dieses Versuchsprojekt enthält bislang nur folgende Wissensquellen:
- **EBM** – Einheitlicher Bewertungsmaßstab
- **Priscus-Liste** – Potenziell inadäquate Medikamente für ältere Patienten
- **FORTA-Liste** – Medikamenten-Risikokategorien A–D für ältere Patienten
- **KBV-Infothek** und weitere Praxisinformationen
""")

st.divider()

frage = st.text_input("🔍 Ihre Frage:", placeholder="z.B. Welche Risiken hat Venlafaxin bei älteren Patienten?")
if st.button("Suchen") and frage:
    with st.spinner("Suche läuft..."):
        antwort = run_smart_query_streamlit(frage)
        st.markdown("### Antwort:")
        st.markdown(antwort)

        # Protokoll
        if "protokoll" not in st.session_state:
            st.session_state.protokoll = []
        st.session_state.protokoll.append({"frage": frage, "antwort": antwort})

# Protokoll anzeigen
if "protokoll" in st.session_state and st.session_state.protokoll:
    st.divider()
    st.markdown("### 📋 Bisherige Abfragen")
    for i, eintrag in enumerate(reversed(st.session_state.protokoll)):
        with st.expander(f"Frage {len(st.session_state.protokoll)-i}: {eintrag['frage']}"):
            st.markdown(eintrag["antwort"])

st.divider()
st.markdown("### 📧 Kontakt")
with st.form("kontakt_formular"):
    name = st.text_input("Ihr Name:")
    email = st.text_input("Ihre E-Mail:")
    nachricht = st.text_area("Ihre Nachricht:")
    absenden = st.form_submit_button("Absenden")
    
    if absenden and name and email and nachricht:
        import requests
        response = requests.post(
            "https://formspree.io/f/xbdqvbnq",
            data={"name": name, "email": email, "message": nachricht}
        )
        if response.status_code == 200:
            st.success("✅ Nachricht erfolgreich gesendet!")
        else:
            st.error("❌ Fehler beim Senden.")
