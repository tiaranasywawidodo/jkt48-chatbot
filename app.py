import streamlit as st
import json
import random
from chat import predict_class

# Load dataset
try:
    with open('intents.json', 'r', encoding='utf-8') as file:
        intents = json.load(file)
except Exception as e:
    st.error(f"Gagal memuat intents.json: {e}")

# --- CONFIGURATION ---
st.set_page_config(page_title="JKT48 Intelligence", page_icon="💎", layout="wide")

# --- CSS: FIXED BOX & NAVY THEME ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

    .stApp {
        background-color: #020617;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    header, footer, #MainMenu {visibility: hidden;}

    .block-container {
        max-width: 800px !important;
        padding: 20px !important;
    }

    /* Area Chat yang Dibatasi (Fixed Height & Scrollable) */
    .chat-fixed-container {
        height: 500px; /* Batas tinggi kotak chat */
        overflow-y: auto; /* Aktifkan scroll hanya di sini */
        padding: 20px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        display: flex;
        flex-direction: column;
    }

    /* Custom Scrollbar agar lebih cantik */
    .chat-fixed-container::-webkit-scrollbar { width: 5px; }
    .chat-fixed-container::-webkit-scrollbar-thumb { background: #1e3a8a; border-radius: 10px; }

    /* Bubble Chat Styling */
    .user-style {
        background: #2563eb;
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 2px 18px;
        align-self: flex-end; /* Paksa ke kanan */
        margin-bottom: 15px;
        font-size: 16px;
        max-width: 80%;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.2);
    }

    .bot-style {
        background: #1e293b;
        color: #f1f5f9;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 2px;
        align-self: flex-start; /* Paksa ke kiri */
        margin-bottom: 15px;
        font-size: 16px;
        max-width: 80%;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Input Styling */
    .stChatInputContainer {
        padding: 20px 0 !important;
        background-color: transparent !important;
    }
    
    .stChatInputContainer textarea {
        background: #0f172a !important;
        color: white !important;
        border: 1px solid #1e3a8a !important;
        border-radius: 15px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
    <div style="text-align: center; padding-bottom: 20px;">
        <h1 style="color: white; font-weight: 800; margin: 0;">JKT48 Assistant</h1>
        <p style="color: #38bdf8; font-size: 12px; letter-spacing: 2px;">JKT48 Information Center</p>
    </div>
""", unsafe_allow_html=True)

# --- CHAT AREA ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Memulai Kotak Chat yang Dibatasi
chat_html = '<div class="chat-fixed-container">'
for msg in st.session_state.messages:
    if msg["role"] == "user":
        chat_html += f'<div class="user-style">{msg["content"]}</div>'
    else:
        chat_html += f'<div class="bot-style">{msg["content"]}</div>'
chat_html += '</div>'

# Menampilkan Kotak Chat
st.markdown(chat_html, unsafe_allow_html=True)

# Input Chat (Di luar kotak scroll agar tetap diam di bawah)
if prompt := st.chat_input("Tanyakan sesuatu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    results = predict_class(prompt)
    if results:
        tag = results[0]['intent']
        response = "Maaf, Rin belum mengerti."
        for i in intents['intents']:
            if i['tag'] == tag:
                response = random.choice(i['responses'])
    else:
        response = "Maaf, data tidak ditemukan."

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()