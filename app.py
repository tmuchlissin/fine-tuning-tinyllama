# app.py
import streamlit as st
import requests
import os

# Set page config
st.set_page_config(page_title="Indonesian University QA & Summarization", layout="centered")

# Header
st.markdown(
    """
    <h3 style="text-align: center; font-size: 34px; margin-bottom: 20px; padding: 0;">
        🏛️ Indonesian University QA & Summarization
    </h3>
    """,
    unsafe_allow_html=True
)
st.markdown("""
**A specialized LLM assistant fine-tuned on Indonesian university campus locations and facilities.**  
Based on structured knowledge about UI, ITB, UGM, IPB, Unair, and ITS.
""")

# Info & Download Section
with st.expander("ℹ️ About This Model"):
    st.markdown("""
    This application uses a **fine-tuned TinyLlama model** trained specifically on:
    
    - ✅ **Question-Answering** about university campuses in Indonesia
    - ✅ **Text Summarization** of campus descriptions
    
    Training data includes detailed information about:
    - Universitas Indonesia (UI)
    - Institut Teknologi Bandung (ITB)
    - Universitas Gadjah Mada (UGM)
    - IPB University
    - Universitas Airlangga (Unair)
    - Institut Teknologi Sepuluh Nopember (ITS)
    
    The model performs best on queries related to these institutions.
    """)

    st.markdown("---")
    st.markdown("📥 **Download Training Data (JSON)**")
    st.markdown("Use these datasets to understand, reproduce, or retrain the model.")

    qa_path = "data/qa.json"
    sum_path = "data/summarization.json"

    if os.path.exists(qa_path):
        with open(qa_path, "r") as f:
            qa_data = f.read()
        st.download_button(
            label="🔽 Download QA Training Data (JSON)",
            data=qa_data,
            file_name="qa.json",
            mime="application/json"
        )
    else:
        st.warning("❌ `training_qa.json` not found. Please ensure it exists in the `data/` folder.")

    if os.path.exists(sum_path):
        with open(sum_path, "r") as f:
            sum_data = f.read()
        st.download_button(
            label="🔽 Download Summarization Training Data (JSON)",
            data=sum_data,
            file_name="summarization.json",
            mime="application/json"
        )
    else:
        st.warning("❌ `training_summarization.json` not found. Please ensure it exists in the `data/` folder.")

# FastAPI base URL
API_URL = "http://127.0.0.1:8000"

# Tabs
tab1, tab2 = st.tabs(["❓ Ask a Question", "📝 Summarize Text"])

# --- QA Tab ---
with tab1:
    st.header("Ask About Indonesian Universities")

    example_questions = [
        "Where is the Jatinangor campus of ITB located?",
        "What city is home to Universitas Gadjah Mada (UGM)?",
        "In which city can I find Universitas Airlangga (Unair)?",
        "Does Universitas Indonesia have a campus in Jakarta?",
        "Where is the main campus of IPB University located?",
        "What is the location of the main campus of ITS?",
        "Which faculties are at the UI Salemba campus?"
    ]

    selected_example = st.selectbox(
        "Choose an example question:",
        options=["Custom Question"] + example_questions,
        index=0
    )

    if selected_example != "Custom Question":
        question = st.text_area("Your Question:", value=selected_example, height=100, key="qa_input")
    else:
        question = st.text_area("Or write your own question:", value="", height=100, key="qa_input")

    if st.button("Get Answer", key="qa_btn"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("🧠 Thinking..."):
                try:
                    response = requests.post(f"{API_URL}/qa", json={"question": question})
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Answer (in {data['response_time']}s):")
                        st.write(data["answer"])
                    else:
                        st.error(f"❌ Error: {response.json().get('detail')}")
                except requests.ConnectionError:
                    st.error("❌ Could not connect to FastAPI server. Make sure it's running with:\n\n`uvicorn main:app --reload`")

# --- Summarization Tab ---
with tab2:
    st.header("Summarize University Information")

    example_texts = {
        "ITB Campus Info": (
            "Institut Teknologi Bandung (ITB) features its main and most historic campus on Jalan Ganesha, Bandung, West Java. "
            "To accommodate growth, it has expanded with a secondary campus in Jatinangor, Sumedang Regency, which houses several of a newer faculties and programs."
        ),
        "Unair Campus Info": (
            "Universitas Airlangga (Unair) is a prominent university located in Surabaya, East Java. Its facilities are spread across three main campus locations within the city: Campus A for Medicine, Campus B for Social Sciences, and Campus C for Science and Technology."
        ),
        "IPB University Info": (
            "Institut Pertanian Bogor (IPB University) has its primary campus in Dramaga, Bogor Regency, West Java. "
            "It also operates a secondary campus in Baranangsiang, Bogor City, which hosts its Faculty of Economics and Business."
        )
    }

    selected_text = st.selectbox(
        "Or choose a pre-defined text:",
        options=["Write your own"] + list(example_texts.keys())
    )

    if selected_text != "Write your own":
        text_input = st.text_area("Text to summarize:", value=example_texts[selected_text], height=200, key="sum_input")
    else:
        text_input = st.text_area("Enter text to summarize:", value="", height=200, key="sum_input")

    if st.button("Summarize", key="sum_btn"):
        if not text_input.strip():
            st.warning("Please enter some text to summarize.")
        else:
            with st.spinner("📝 Generating summary..."):
                try:
                    response = requests.post(f"{API_URL}/summarize", json={"text": text_input})
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Summary (in {data['response_time']}s):")
                        st.write(data["summary"])
                    else:
                        st.error(f"❌ Error: {response.json().get('detail')}")
                except requests.ConnectionError:
                    st.error("❌ Could not connect to FastAPI server. Make sure it's running.")

# Footer
st.markdown("---")
st.markdown("🔍 **Tip**: The model works best on universities in the training data. Results on unrelated topics may be generic or inaccurate.")