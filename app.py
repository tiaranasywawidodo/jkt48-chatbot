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

# --- CSS: FIXED HEADER & SCROLLABLE CHAT ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

    .stApp {
        background-color: #020617;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    header, footer, #MainMenu {visibility: hidden;}

    /* Container utama aplikasi */
    .block-container {
        max-width: 800px !important;
        padding-top: 120px !important; /* Jarak agar tidak tertutup header */
        padding-bottom: 100px !important;
    }

    /* --- HEADER FIX (DIAM DI ATAS) --- */
    .my-fixed-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background: rgba(2, 6, 23, 0.95);
        backdrop-filter: blur(10px);
        z-index: 999;
        text-align: center;
        padding: 20px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* --- CHAT AREA --- */
    .chat-wrapper {
        display: flex;
        flex-direction: column;
        gap: 15px;
    }

    .user-style {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 2px 18px;
        align-self: flex-end;
        margin-left: auto;
        font-size: 16px;
        max-width: 80%;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.2);
        animation: fadeInRight 0.3s ease;
    }

    .bot-style {
        background: #1e293b;
        color: #f1f5f9;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 2px;
        align-self: flex-start;
        margin-right: auto;
        font-size: 16px;
        max-width: 80%;
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: fadeInLeft 0.3s ease;
    }

    @keyframes fadeInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes fadeInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    </style>

    <div class="my-fixed-header">
        <h1 style="color: white; font-weight: 800; margin: 0; font-size: 28px;">JKT48 Assistant</h1>
        <p style="color: #38bdf8; font-size: 11px; letter-spacing: 2px; margin: 0; text-transform: uppercase;">JKT48 Information Center</p>
    </div>
""", unsafe_allow_html=True)

# --- CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan chat dalam container div
chat_placeholder = st.container()

with chat_placeholder:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-style">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-style">{msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Input Chat
if prompt := st.chat_input("Tanyakan sesuatu..."):
    # Simpan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Prediksi respon
    results = predict_class(prompt)
    if results:
        tag = results[0]['intent']
        response = "Maaf, Rin belum mengerti."
        for i in intents['intents']:
            if i['tag'] == tag:
                response = random.choice(i['responses'])
    else:
        response = "Maaf, data tidak ditemukan."

    # Simpan pesan bot
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()