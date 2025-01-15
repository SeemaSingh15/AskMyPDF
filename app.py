import os
import streamlit as st
from dotenv import load_dotenv  # You can remove this since we won't use .env anymore
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None


def get_text_chunks(text):
    if not text:
        return None

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    if not text_chunks:
        return None

    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None


def get_conversation_chain(vectorstore):
    if not vectorstore:
        return None

    try:
        # Access the Hugging Face token from Streamlit's secrets
        api_token = st.secrets["huggingface"]["api_token"]
        if not api_token:
            st.error("Missing HUGGINGFACE_API_TOKEN in secrets.toml")
            return None

        # Create the HuggingFace model
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={
                "temperature": 0.5,
                "max_length": 512,
                "top_p": 0.9,
            },
            huggingfacehub_api_token=api_token
        )

        # Set up memory for the conversation
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        # Create the conversation chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            verbose=True
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.error("Please upload and process a PDF document first!")
        return

    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")


def main():
    # No need to load .env anymore as we are using secrets.toml for Hugging Face token
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Chat with multiple PDFs :books:")

    if st.session_state.conversation is None:
        st.warning("👋 Please upload your PDF documents in the sidebar and click 'Process' to start!")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type=['pdf']  # Only allow PDF files
        )

        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file!")
                return

            with st.spinner("Processing your PDFs..."):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    # Get text chunks
                    text_chunks = get_text_chunks(raw_text)
                    if text_chunks:
                        # Create vector store
                        vectorstore = get_vectorstore(text_chunks)
                        if vectorstore:
                            # Create conversation chain
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            if st.session_state.conversation:
                                st.success("✅ Processing complete! You can now ask questions about your documents.")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)


if __name__ == '__main__':
    main()
