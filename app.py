'''
latest version(25/08/24)
user sessions working(not interfaring with different users)
headings are working fine now
'''
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import os
import shutil
from get_embedding_function import get_embedding_function
import populate_database
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import base64

CHROMA_PATH = "chroma"
DATA_PATH = "PDFs"
model_name = "llama3"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str, username):
    # embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH+"_"+username, embedding_function=get_embedding_function())
    db.persist()
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = Ollama(model=model_name)
    response_text = model.invoke(prompt)
    return response_text

def handle_pdf_upload(uploaded_file, username, reset = True):
    print("\n\nHandlePDF Called!\n\n")
    if os.path.exists(DATA_PATH+"/"+username):
        shutil.rmtree(DATA_PATH+"/"+username)
    os.makedirs(DATA_PATH+"/"+username)
    save_path = os.path.join(DATA_PATH+"/"+username, uploaded_file.name)
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.session_state['pdf_uploaded'] = True
    st.sidebar.success(f"File saved to {save_path}")
    
    populate_database.main(reset, username = username)

def get_base64_of_image(image_file):
    with open(image_file, "rb") as image:
        return base64.b64encode(image.read()).decode()

def main():
    st.set_page_config(page_title="Q&A with LLM", page_icon="💬", layout="wide")

    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)



    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    name, authentication_status, username = authenticator.login()

    # if authentication_status:
    #     authenticator.logout('Logout', 'main')
    #     st.write(f'Welcome *{name}*')
    #     st.title('Some content')
    # elif authentication_status == False:
    #     st.error('Username/password is incorrect')
    # elif authentication_status == None:
    #     st.warning('Please enter your username and password')

    if st.session_state["authentication_status"]:
        
        st.markdown("""
        <style>
            .title-container {
                display: flex;
                position: fixed;
                top: 0;
                margin-top: 40px;
                align-items: center;
                background: white;
                padding: 0px;
                width :100%;
                border-bottom: 1px solid #e6e6e6;
            }
            .text-container {
                flex: 1;
                text-align: left;
                margin-left: 10px;
                
            }
            @media (prefers-color-scheme: dark) {
                .title-container {
                    background: transparent;
                    color: white;
                    border-bottom: 1px solid #555555;
                }
            }
        </style>
        """, 
        unsafe_allow_html=True)

        st.sidebar.title("Upload PDF")
        uploaded_file = st.sidebar.file_uploader("", type="pdf")
        
        if uploaded_file is not None:
            if "fileUpload" not in st.session_state:
                # st.session_state.fileUpload = uploaded_file
                st.session_state.fileUpload = uploaded_file.name
                print("first file fo session uploaded")
                handle_pdf_upload(uploaded_file, username, reset = True)
            else:
                if st.session_state.fileUpload!=uploaded_file.name:
                    print("new file uploaded ")
                    st.session_state.fileUpload=uploaded_file.name
                    handle_pdf_upload(uploaded_file, username, reset=True)
                else:print("Same file uploaded again")

        st.markdown("""
                    <div class='title-container'>
                    <div class='text-container'>
                        <h1>Conversational bot 💬</h1>
                    </div>
                </div>
""", unsafe_allow_html=True)
        authenticator.logout(location="sidebar")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask your query"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Fetching response..."):
                response = query_rag(prompt, username=username)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

    elif st.session_state["authentication_status"] == False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] == None:
        st.warning('Please enter your username and password')



    
if __name__ == "__main__":
    main()
