#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:33:45 2026

@author: sigillus
"""
#==================================================
# hybrid RAG  bot mit reciprocal ranking
# HIER IN ANBINDUNG AN  SQLITE  über huggingface
#==================================================

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
from huggingface_hub import hf_hub_download
import streamlit as st
from dotenv import load_dotenv

try:
    qui = os.environ.get("TOGETHER_API_KEY")
except: 
    load_dotenv()
    qui = os.environ.get("TOGETHER_API_KEY")

st.info("⬇️ Lade Datenbank aus dem Netz, bitte ca.20 Sekunden Geduld")
DB_PATH = "medical_data.db" 
try: 
  if not os.path.exists(DB_PATH):
    downloaded = hf_hub_download(
        repo_id="Sigillus/medic_Chat",
        filename="medical_data.db",
        repo_type="dataset",
        local_dir="."
    )
    DB_PATH = downloaded
except Exception as e:
   st.error(f"❌ Download fehlgeschlagen: {e}")
   st.stop()
    
db_connection_str = f"sqlite:///{DB_PATH}"

# --- 2. TEXTE AUS DB LADEN (für BM25) ---
@st.cache_resource
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
    """
    Liest gespeicherte Embedding-Strings direkt aus SQLite –
    kein Neuberechnen, kein externes Paket, nur numpy.
    """
    def __init__(self, docs, vectors, embeddings_model):
        import numpy as np
        self.docs = docs
        self.embeddings_model = embeddings_model
        self.matrix = np.array(vectors, dtype="float32")
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.matrix = self.matrix / np.maximum(norms, 1e-10)

    def invoke(self, query, k=12):
        import numpy as np
        q_vec = np.array(self.embeddings_model.embed_query(query), dtype="float32")
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-10)
        scores = self.matrix @ q_vec
        indices = np.argsort(scores)[::-1][:int(k)]
        return [self.docs[i] for i in indices]

@st.cache_resource
def load_vectorstore_from_sqlite(db_path, embeddings_model):
    """
    Lädt Dokumente + bereits gespeicherte Embedding-Vektoren aus SQLite.
    Die Embeddings liegen als JSON-String vor z.B. '[-0.012, 0.024, ...]'
    """
    import json
    from langchain_core.documents import Document

    docs = []
    vectors = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT document, cmetadata, embedding FROM "langchain_embedding" '
                'WHERE document IS NOT NULL AND embedding IS NOT NULL'
            )
            for row in cursor.fetchall():
                text, meta_str, emb_str = row
                metadata = {}
                try:
                    metadata = json.loads(meta_str) if meta_str else {}
                except Exception:
                    pass
                try:
                    vec = json.loads(emb_str)
                except Exception:
                    continue  # Zeile überspringen wenn Vektor nicht lesbar
                docs.append(Document(page_content=text, metadata=metadata))
                vectors.append(vec)

    except Exception as e:
        print(f"Fehler beim Laden des Vectorstores: {e}")

    if docs:
        print(f"✅ {len(docs)} Dokumente mit Vektoren geladen")
        return SimpleVectorRetriever(docs, vectors, embeddings_model)
    print("⚠️ Keine Dokumente geladen")
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
    "die 'FORTA-Liste' enthält Risikokategorien für Medikamenten-Gaben an ältere Patienten von A (gut) bis D (sehr gefährlich).\n\n"
    "'ebm': Enthält Abrechungsziffern und Ausschlüsse zur Abrechnung von Patienten in Praxen.\n\n"
    "'gelbe Liste': enthält eine Zuordnung von \n\n"
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
        #return(f"⚠️ Keine Treffer in der Datenbank.")
        return {"antwort": "⚠️ Keine Treffer in der Datenbank.", "quellen": ""}
    
    current_chain = prompt | model
         
    try:
        result = current_chain.invoke({
        "context": data["context"],
        "question": question
        })
        return  {
            "antwort": result.content, 
            "quellen": data["sources"]
        } 
    except Exception as e:
      return {
            "antwort": f"Ein Fehler ist aufgetreten: {str(e)}", 
            "quellen": ""
        }

st.set_page_config(
    page_title="ChatBot für praktische Ärzte",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Prototypischer ChatBot mit Sonderwissen für praktische Ärzte")

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

st.markdown('<p style="font-size:24px; font-weight:bold;">🔍 Ihre Frage:</p>', unsafe_allow_html=True)
frage = st.text_input(
    label="🔍  Ihre Frage", 
    placeholder="z.B. Welche Risiken hat Venlafaxin bei älteren Patienten? (überschreiben und <SUCHEN> klicken)",
    label_visibility="collapsed"
)

# Eingabefeld  und  Checkbox für die Quellen-Abfrage
st.markdown("---")
quellen_anzeigen = st.checkbox("Quellen anzeigen?")

if st.button("Suchen") and frage:
    with st.spinner("Suche läuft..."):
        # Ergebnis abrufen (jetzt ein Dictionary)
        ergebnis = run_smart_query_streamlit(frage)
        antwort = ergebnis["antwort"]
        quellen = ergebnis["quellen"]
        
        # Antwort anzeigen
        st.markdown("### Antwort:")
        st.markdown(antwort)
        
        # Logik für die Quellenanzeige
        if quellen_anzeigen and quellen:
            with st.expander("📚 Verwendete Quellen", expanded=True):
                st.markdown(quellen)
        
        # Protokoll speichern (mit Quellen für später)
        if "protokoll" not in st.session_state:
            st.session_state.protokoll = []
        st.session_state.protokoll.append({
            "frage": frage, 
            "antwort": antwort, 
            "quellen": quellen
        })
        
st.divider()
st.markdown("### like/dislike? 📧 drop a note to Helge")

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
