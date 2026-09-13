from __future__ import annotations
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from fmea_schema import empty_fmea_df, normalize_scores
from llm import LLMClient, LLMConfig
from fmea_chatbot import answer_question
from utils import to_csv_bytes, to_xlsx_bytes

load_dotenv()

st.set_page_config(page_title="Medical-Device-Risk-Analyzer (CT Scan)", layout="wide")
st.title("Medical-Device-Risk-Analyzer — CT Scan")

# -------- Session State --------
if "fmea_df" not in st.session_state:
    st.session_state.fmea_df = empty_fmea_df()
if "fmea_saved" not in st.session_state:
    st.session_state.fmea_saved = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------- Sidebar: LLM Config --------
st.sidebar.header("Config")

model = st.sidebar.text_input(
    "Qwen Model",
    value="qwen-turbo"
)

llm = LLMClient(
    cfg=LLMConfig(model=model, provider="qwen"),
    api_key=os.getenv("QWEN_API_KEY")
)

tab1, tab2 = st.tabs(["FMEA Builder (CT Scan)", "Chatbot (QnA FMEA)"])

# -------- TAB 1: FMEA Builder --------
with tab1:
    st.subheader("1) Build / Load FMEA Table")

    colA, colB = st.columns([2, 1], gap="large")

    # ---- LEFT: Manual Edit ----
    with colA:
        st.markdown("### Edit Table")
        edited = st.data_editor(
            st.session_state.fmea_df,
            num_rows="dynamic",
            use_container_width=True,
            key="fmea_editor",
        )
        st.session_state.fmea_df = normalize_scores(edited)

    # ---- RIGHT: Load / Save ----
    with colB:
        st.markdown("### Upload Existing FMEA CSV (optional)")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded:
            st.session_state.fmea_df = pd.read_csv(uploaded)
            st.success("FMEA loaded from CSV!")

        st.markdown("### Save / Download")
        csv_bytes = to_csv_bytes(st.session_state.fmea_df)
        xlsx_bytes = to_xlsx_bytes(st.session_state.fmea_df)

        st.download_button("Download CSV", data=csv_bytes,
                           file_name="FMEA_CT.csv", mime="text/csv")
        st.download_button("Download XLSX", data=xlsx_bytes,
                           file_name="FMEA_CT.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        if st.button("Mark as saved (enable chatbot)"):
            st.session_state.fmea_saved = True
            st.success("Saved! Chatbot is enabled.")

# -------- TAB 2: Chatbot --------
with tab2:
    st.subheader("2) Chatbot — Ask about the saved FMEA")

    if not st.session_state.fmea_saved:
        st.warning("Save tabel FMEA dulu sebelum pakai chatbot.")
    else:
        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(msg)

        q = st.chat_input("Ask something about your FMEA table…")
        if q:
            st.session_state.chat_history.append(("user", q))
            with st.chat_message("assistant"):
                ans = answer_question(llm, st.session_state.fmea_df, q)
                st.write(ans)
            st.session_state.chat_history.append(("assistant", ans))

#raisa2025