import streamlit as st
import requests
import json
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/api")

st.set_page_config(page_title="DocRAG", page_icon="📚", layout="wide")
st.title("📚 DocRAG: Document Q&A")

# Sidebar
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or Text files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing..."):
                files_data = [
                    ("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files
                ]
                try:
                    resp = requests.post(f"{API_URL}/upload", files=files_data)
                    if resp.status_code == 200:
                        st.success("Documents processed!")
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
        else:
            st.warning("Please upload files first.")

    st.markdown("---")
    st.header("Settings")
    google_key = st.text_input("Google API Key (Optional)", type="password")

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {"question": prompt}
    if google_key:
        payload["google_api_key"] = google_key

    with st.chat_message("assistant"):
        container = st.empty()
        full_answer = ""

        try:
            resp = requests.post(f"{API_URL}/chat", json=payload, stream=True, timeout=60)
            if resp.status_code == 200:
                for line in resp.iter_lines():
                    if line:
                        decoded = line.decode("utf-8")
                        if decoded.startswith("data: "):
                            content = decoded[6:]
                            if content == "[DONE]":
                                break
                            try:
                                data = json.loads(content)
                                if data["type"] == "answer":
                                    full_answer += data["content"]
                                    container.markdown(full_answer + "▌")
                                elif data["type"] == "error":
                                    st.error(f"Backend error: {data['content']}")
                            except json.JSONDecodeError:
                                continue

                container.markdown(full_answer if full_answer else "No answer received.")
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_answer}
                )
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Is it running?")
        except Exception as e:
            st.error(f"Error: {e}")
