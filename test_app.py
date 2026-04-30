#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:33:45 2026

@author: sigillus
"""
#=============================================
# hybrid RAG  bot mit reciprocal ranking
# HIER IN ANBINDUNG AN S Q L I T E
#========================================

import os
import sqlite3
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import SQLiteVSS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import streamlit as st
from dotenv import load_dotenv

#load_dotenv()
try:
 qui = os.environ.get("TOGETHER_API_KEY")
except: 
 load_dotenv()
 qui = os.environ.get("TOGETHER_API_KEY")

# --- SQLite Pfad ---
DB_PATH = os.path.join(os.path.dirname(__file__), 'medical_data.db')
db_connection_str = f"sqlite:///{DB_PATH}"


# --- 2. TEXTE AUS DB LADEN (für BM25) ---
def get_all_texts_from_db(db_path):
    texts = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Alle Tabellen holen, die NICHT zu langchain gehören
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table'
                AND name NOT LIKE 'langchain_%'
                AND name NOT LIKE 'sqlite_%'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                # Text-Spalten in jeder Tabelle finden
                cursor.execute(f"PRAGMA table_info('{table}')")
                columns_info = cursor.fetchall()
                text_columns = [
                    row[1] for row in columns_info
                    if row[2].upper() in ('TEXT', 'VARCHAR', 'CHAR', '')
                ]
                
                for col in text_columns:
                    try:
                        cursor.execute(f'SELECT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL')
                        rows = cursor.fetchall()
                        texts.extend([row[0] for row in rows if isinstance(row[0], str)])
                    except Exception:
                        pass
                        
    except Exception as e:
        print(f"Fehler beim Laden der Texte: {e}")
    return texts

all_texts = get_all_texts_from_db(DB_PATH)


# --- 3. MODELLE & EMBEDDINGS ---
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
# Vektoren direkt aus SQLite laden (LangChain speichert document + embedding)
class SimpleVectorRetriever:
    """Einfacher In-Memory Vectorstore ohne externe Abhängigkeiten – nur numpy."""
    def __init__(self, docs, embeddings):
        import numpy as np
        self.docs = docs
        self.embeddings = embeddings
        texts = [d.page_content for d in docs]
        vecs = embeddings.embed_documents(texts)
        self.matrix = np.array(vecs, dtype="float32")
        # Normalisieren für Cosine Similarity
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.matrix = self.matrix / np.maximum(norms, 1e-10)

    def invoke(self, query, k=12):
        import numpy as np
        q_vec = np.array(self.embeddings.embed_query(query), dtype="float32")
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-10)
        scores = self.matrix @ q_vec
        top_k = int(k)
        indices = np.argsort(scores)[::-1][:top_k]
        return [self.docs[i] for i in indices]


def load_vectorstore_from_sqlite(db_path, embeddings):
    """Lädt gespeicherte Dokumente aus SQLite und baut einen In-Memory Vectorstore."""
    import json
    from langchain_core.documents import Document

    docs = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name LIKE 'langchain_%'
            """)
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                cursor.execute(f"PRAGMA table_info('{table}')")
                cols = [row[1] for row in cursor.fetchall()]

                if 'document' in cols:
                    meta_col = 'cmetadata' if 'cmetadata' in cols else None
                    if meta_col:
                        cursor.execute(f'SELECT document, {meta_col} FROM "{table}"')
                    else:
                        cursor.execute(f'SELECT document FROM "{table}"')

                    for row in cursor.fetchall():
                        text = row[0]
                        metadata = {}
                        if meta_col and row[1]:
                            try:
                                metadata = json.loads(row[1])
                            except Exception:
                                pass
                        if text:
                            docs.append(Document(page_content=text, metadata=metadata))
    except Exception as e:
        print(f"Fehler beim Laden des Vectorstores: {e}")

    if docs:
        return SimpleVectorRetriever(docs, embeddings)
    return None

vector_retriever = load_vectorstore_from_sqlite(DB_PATH, embeddings)

bm25_retriever = None
if all_texts:
    bm25_retriever = BM25Retriever.from_texts(all_texts)
    bm25_retriever.k = 12

db = SQLDatabase.from_uri(db_connection_str)
sql_toolkit = SQLDatabaseToolkit(db=db, llm=model)
sql_tools = sql_toolkit.get_tools()

# --- 5. HILFSFUNKTIONEN FÜR DIE SUCHE ---

def hybrid_retrieval_with_sources(query, k=60):
    """Kombiniert Vektor- und BM25-Suche mittels Reciprocal Rank Fusion."""
    # Bereinigung der Query
    query = query.replace("Welche Risiken hat ", "").replace("?", "").strip()
    
    v_docs = vector_retriever.invoke(query) if vector_retriever else []
    b_docs = bm25_retriever.invoke(query) if bm25_retriever else []
    
    # RRF Score Berechnung
    doc_scores = {}
    
    for rank, doc in enumerate(v_docs, start=1):
        doc_scores[doc.page_content] = {"score": 1 / (rank + k), "doc": doc}
        
    for rank, doc in enumerate(b_docs, start=1):
        if doc.page_content in doc_scores:
            doc_scores[doc.page_content]["score"] += 1 / (rank + k)
        else:
            doc_scores[doc.page_content] = {"score": 1 / (rank + k), "doc": doc}

    # Nach Score sortieren
    reranked_results = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)

    context_parts = []
    sources = set()

    # SQL-Suche in unvektorisierten Tabellen (SQLite: LIKE statt ILIKE)
    try:
        sql_result = db.run(f"SELECT * FROM priscus WHERE wirkstoff LIKE '%{query}%' LIMIT 5;")
        if sql_result:
            context_parts.append(f"Priscus-Liste:\n{sql_result}")
    except:
        pass

    try:
        sql_result = db.run(f"SELECT * FROM forta_liste WHERE wirkstoff LIKE '%{query}%' LIMIT 5;")
        if sql_result:
            context_parts.append(f"FORTA-Liste:\n{sql_result}")
    except:
        pass

    for item in reranked_results:
        doc = item["doc"]
        context_parts.append(doc.page_content)
        
        source_name = doc.metadata.get("source", "Unbekannte Quelle")
        page_num = doc.metadata.get("page", "?")
        source_file = os.path.basename(source_name)
        sources.add(f"- {source_file} ")
    
    return {
        "context": "\n\n---\n\n".join(context_parts),
        "sources": "\n".join(sources)
    }

# --- 6. PROMPT SETUP ---
instructions = (
    "Du bist ein spezialisierter medizinischer Experte für Medikamentensicherheit und Praxisführung"
    "Deine Wissensbasis sind AUSSCHLIESSLICH die bereitgestellten Textstellen"
    "Deine einzige Aufgabe ist es, die Frage auf Basis des unten gelieferten KONTEXTES zu beantworten.\n\n"
    "INFORMATIONEN ZUR STRUKTUR:\n"
    "im Dokumentt 'priscus-liste' sind Probleme zu Wirkstoffen und Medikamenten bei älteren Patienten enthalten"
    "die 'FORTA-Liste': Enthält Risikokategorien für Medikamenten-Gaben an ältere Patienten von A (gut) bis D (sehr gefährlich).\n\n"
    "'ebm': Enthält Abrechungsziffern und Ausschlüsse zur Abrechnung von Patienten in Praxen.\n\n"
    "DEINE REGELN:\n"
    "2. Danach nutze 'pdf_semantic_search' \n"
    "3. Nutze bei Abfragen IMMER 'LIKE %...%' für Medikamentennamen, da diese unterschiedlich geschrieben sein können (z.B. WHERE name LIKE '%Amlodipin%').\n"
    "5. Wenn du keine Informationen findest, antworte: 'Ich konnte keine Informationen zu [Medikament] in der Datenbank finden.'\n"
    "6. Antworte NUR auf Basis der gefundenen Texte. Wenn nichts gefunden wird, sag das.\n"
    "7. Gib NIEMALS SQL-Beispiele oder allgemeines Wissen aus."
    "8. Erfinde keine medizinischen Fakten und nenne keine Tool-Namen oder SQL-Befehle.\n"
    "10. Erfinde niemals Wirkmechanismen oder vergleiche Medikamente aus deinem eigenen Training."
)

prompt = ChatPromptTemplate.from_template(
    instructions + "\n\nKONTEXT:\n{context}\n\nFRAGE: {question}\n\nANTWORT:"
)

# --- 7. HAUPTFUNKTION ---

def run_smart_query_streamlit(question):
    print(f"\n🔎 Suche läuft für: {question}...")
    
    # 1. Daten abrufen
    data = hybrid_retrieval_with_sources(question)
    
    if not data["context"]:
        return(f"⚠️ Keine Treffer in der Datenbank.")
    
    current_chain = prompt | model
         
    try:
        result = current_chain.invoke({
        "context": data["context"],
        "question": question
        })
        return result.content
    except Exception as e:
      return f"Fehler: {e}"


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
- **KBV-Infothek: Wirkstoff aktuell** – Aktuelle Wirkstoffinformationen der kassenärztlichen Bundesvereinigung
- **KBV-Infothek: Praxiswissen und KBV-Praxisinformationen**   – Praxisrelevante Informationen 
- **Teile der Gelbe Liste** – Medikamenteninformationen
- **ICD-Codes** – Diagnoseschlüssel
""")

st.divider()

# Eingabefeld
frage = st.text_input("🔍 Ihre Frage:", placeholder="z.B. Welche Risiken hat Venlafaxin bei älteren Patienten?   (ins Feld klicken und überschreiben)")

if st.button("Suchen") and frage:
    with st.spinner("Suche läuft..."):
        antwort = run_smart_query_streamlit(frage)
        
        # Antwort anzeigen
        st.markdown("### Antwort:")
        st.markdown(antwort)
        
        # Protokoll speichern
        if "protokoll" not in st.session_state:
            st.session_state.protokoll = []
        st.session_state.protokoll.append({"frage": frage, "antwort": antwort})

# Protokoll anzeigen
if "protokoll" in st.session_state and len(st.session_state.protokoll) > 0:
    st.divider()
    st.markdown("### 📋 Bisherige Abfragen")
    for i, eintrag in enumerate(reversed(st.session_state.protokoll)):
        with st.expander(f"Frage {len(st.session_state.protokoll)-i}: {eintrag['frage']}"):
            st.markdown(eintrag["antwort"])
            
st.divider()
st.markdown("### 📧 drop a message to Helge")

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
