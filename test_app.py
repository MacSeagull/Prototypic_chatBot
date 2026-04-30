import streamlit as st
import sqlite3
import pandas as pd

st.title("Medical Data Check")

def load_data():
    conn = sqlite3.connect('medical_data.db')
    query = "SELECT * FROM langchain_embedding LIMIT 5"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

try:
    data = load_data()
    st.write("Erfolgreich verbunden! Erste 5 Einträge:")
    st.dataframe(data)
except Exception as e:
    st.error(f"Fehler: {e}")
    st.info("Prüfe ob 'medical_data.db' im gleichen Ordner liegt.")
