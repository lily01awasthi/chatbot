import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from huggingface_hub import InferenceClient

# Changes to the depricated model
inference = InferenceClient()



def get_conversational_chain(vector_stores):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_stores.as_retriever(),
        memory=memory
    )
    return conversational_chain


# Define a function to display a PDF in the browser
def get_text_from_pdfs(uploaded_files):
    text = ""
    for file in uploaded_files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_data):
    char_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = char_splitter.split_text(raw_data)
    return chunks


def get_vector_stores(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_stores = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_stores


def sidebar():
    load_dotenv()
    if "conversation" not in st.session_state:
        conversation = None
    if "chat_history" not in st.session_state:
        chat_history = None
    with st.sidebar:
        # Sidebar for user input
        st.sidebar.header('User Input Options')
        uploaded_files = st.sidebar.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

        if st.button("process"):
            with st.spinner("processing"):
                # get pdf texts
                raw_data = get_text_from_pdfs(uploaded_files)

                # get text chunks
                text_chunks = get_text_chunks(raw_data)

                # get vector stores
                vector_stores = get_vector_stores(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversational_chain(vector_stores)


def handle_user_input(user_query):
    response = st.session_state.conversation({'question': user_query})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(message.content)  # add user template
        else:
            st.write(message.content) #add bot template

def layout():
    # Set the title and the layout
    st.set_page_config(page_title='PDF Upload and Query App', layout="wide")
    st.title('PDF Upload and Query App')
    user_query = st.text_input("Enter your query here")
    if user_query:
        handle_user_input(user_query)


if __name__ == '__main__':
    layout()
    sidebar()
