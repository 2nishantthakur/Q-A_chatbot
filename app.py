import os
import shutil
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
import populate_database
from chromadb.config import Settings

CHROMA_PATH = "chroma"
DATA_PATH = "PDFs"
MODEL_NAME = "llama3"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function, client_settings=Settings())

    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model=MODEL_NAME)
    response_text = model.invoke(prompt)

    return response_text

def handle_pdf_upload(uploaded_file):
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)
    os.makedirs(DATA_PATH)

    save_path = os.path.join(DATA_PATH, uploaded_file.name)
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.session_state['pdf_uploaded'] = True
    st.success(f"File saved to {save_path}")

    populate_database.main(reset=True)

def main():
    st.set_page_config(page_title="Q&A with LLM", page_icon="💬", layout="wide")

    st.markdown(
        """
        <style>
            .main {
                max-width: 100%;
                overflow-x: auto;
            }
            .content {
                display: flex;
                flex-direction: column;
                width: 2000px;  /* Adjust as needed to exceed viewport width */
            }
            .stTextArea, .stButton {
                text-align: center;
                margin: auto;
            }
            .highlight {
                font-weight: bold;
                font-size: 1.1em;
                color: #4CAF50;
            }
            .file-uploader {
                text-align: center;
            }
            /* Optional: Ensure that scrollbars are visible if necessary */
            body {
                overflow-x: auto;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Q&A 💬")
    st.markdown("""
        <div style="text-align: center;">
            This application allows you to ask questions to a language model. 
            Enter your query below and click 'Submit' to get an answer based on the context.
        </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='content'>", unsafe_allow_html=True)

        st.markdown("<p class='highlight'>Add PDF</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type="pdf")

        if uploaded_file is not None:
            if "fileUpload" not in st.session_state:
                st.session_state.fileUpload = uploaded_file
                handle_pdf_upload(uploaded_file)
            else:
                if st.session_state.fileUpload != uploaded_file:
                    st.session_state.fileUpload = uploaded_file
                    handle_pdf_upload(uploaded_file)

        with st.form(key="query_form"):
            query_text = st.text_area("Enter your Query (Type /bye to exit):", height=150)
            submit_button = st.form_submit_button(label="Submit")

        if submit_button:
            if query_text:
                if query_text.lower() == "/bye":
                    st.write("Goodbye!")
                else:
                    with st.spinner("Fetching response..."):
                        response = query_rag(query_text)
                    st.markdown(f"<div>{response}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
