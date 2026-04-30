import streamlit as st
import sqlite3
import pandas as pd

st.title("Medical Data Check")

# Verbindung zur SQLite-Datenbank herstellen
# (Stelle sicher, dass medical_data.db im gleichen GitHub-Ordner liegt wie dieses Skript)
def load_data():
    conn = sqlite3.connect('medical_data.db')
    query = "SELECT * FROM langchain_embedding LIMIT 5"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

try:
    data = load_data()
    st.write("Erfolgreich mit der Datenbank verbunden! Hier sind die ersten 5 Einträge:")
    st.dataframe(data)
except Exception as e:
    st.error(f"Fehler beim Laden der Datenbank: {e}")
    st.info("Hinweis: Prüfe, ob die Datei 'medical_data.db' korrekt nach GitHub hochgeladen wurde.")
