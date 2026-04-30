import streamlit as st
import sqlite3
import pandas as pd
import json
import os

st.title("Medical Data Check – Gespeicherte Texte")

def load_texts():
    db_path = os.path.join(os.path.dirname(__file__), 'medical_data.db')
    conn = sqlite3.connect(db_path)
    
    # Erstmal schauen welche Spalten existieren
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(langchain_embedding)")
    columns = [row[1] for row in cursor.fetchall()]
    st.write("Gefundene Spalten:", columns)
    
    # Texte und Metadaten laden (OHNE den Embedding-Vektor)
    cols_to_show = [c for c in columns if c != 'embedding']
    query = f"SELECT {', '.join(cols_to_show)} FROM langchain_embedding LIMIT 20"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def parse_metadata(df):
    # cmetadata ist oft als JSON-String gespeichert
    if 'cmetadata' in df.columns:
        try:
            df['cmetadata'] = df['cmetadata'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        except:
            pass
    return df

try:
    data = load_texts()
    data = parse_metadata(data)
    
    st.success(f"✅ {len(data)} Einträge geladen")
    
    # Originaltext direkt anzeigen
    if 'document' in data.columns:
        st.subheader("Originaltexte:")
        for i, row in data.iterrows():
            with st.expander(f"Eintrag {i+1}"):
                st.write(row['document'])
                if 'cmetadata' in data.columns:
                    st.json(row['cmetadata'])
    else:
        st.dataframe(data)

except Exception as e:
    st.error(f"Fehler: {e}")
    st.info("Prüfe ob 'medical_data.db' im gleichen Ordner liegt.")
