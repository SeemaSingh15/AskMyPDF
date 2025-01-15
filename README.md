# 💬 Chat with Multiple PDFs - Streamlit Application

This project is a simple Streamlit application that allows users to upload multiple PDF documents, process their content, and ask questions based on the PDFs. The application uses various libraries and tools to enable seamless document processing and a chatbot-like interface for interacting with the content.

## 📚 Libraries Used

This project utilizes several libraries to ensure smooth document handling and question answering:

- **Streamlit** (`streamlit==1.41.1`): For creating the interactive frontend, allowing users to upload PDFs and interact with the chatbot.
- **Python Dotenv** (`python-dotenv==1.0.1`): To manage environment variables (though switched to `secrets.toml` for sensitive data management like the Hugging Face API token).
- **PyPDF2** (`PyPDF2==3.0.1`): A Python library used to extract text from uploaded PDF documents.
- **Langchain** (`langchain==0.3.14`): Enables the integration of AI models, document retrieval, and conversational memory.
- **FAISS** (`faiss-cpu==1.9.0.post1`): Efficient similarity search and clustering of document embeddings.
- **Sentence-Transformers** (`sentence-transformers==3.3.1`): Converts text into dense vectors (embeddings), which are then indexed by FAISS.
- **Hugging Face Hub** (`huggingface-hub==0.27.1`): For accessing and using Hugging Face's pre-trained language models.
- **NumPy** (`numpy==2.2.1`): Required for numerical operations used by some of the libraries (like FAISS and Sentence-Transformers).

## 🎯 Project Overview

The main goal of this project is to create an interactive platform where users can:

- **Upload PDF Documents**: Upload multiple PDF files via the Streamlit interface.
- **Process PDFs**: Extract text, split it into chunks, and store it as vectors for fast retrieval.
- **Ask Questions**: Users can interact with the chatbot by asking questions, which are answered based on the content extracted from the PDFs.
- **Chat History**: The app maintains chat history, enabling users to ask follow-up questions based on previous responses.

The application uses **Hugging Face's Flan-T5 model** for generating conversational responses.

## 🚀 How to Run the Application

### 1. Clone the Repository

Clone the project repository to your local machine:

```bash
git clone https://github.com/SeemaSingh15/AskMyPDF.git
cd AskMyPDF
```

### 2. Install Dependencies

Make sure you have Python 3.8+ installed. Then, install the required libraries:

```bash
pip install -r requirements.txt
```

Alternatively, you can install the dependencies manually using pip:

```bash
pip install streamlit==1.41.1
pip install python-dotenv==1.0.1
pip install PyPDF2==3.0.1
pip install langchain==0.3.14
pip install faiss-cpu==1.9.0.post1
pip install sentence-transformers==3.3.1
pip install huggingface-hub==0.27.1
pip install numpy==2.2.1
```

### 3. Set Up Hugging Face API Token

To use the Hugging Face model, you will need an API token. Follow these steps:

1. Go to Hugging Face and create an account (if you don't have one already).
2. Navigate to API tokens and create a new token.
3. Store the token in your secrets.toml file located in the .streamlit folder within your project directory. The file should look like this:

```toml
[huggingface]
api_token = "your_api_token_here"
```

### 4. Run the Streamlit Application

Once the dependencies are installed and the Hugging Face token is set, you can run the application with the following command:

```bash
streamlit run app.py
```

This will start the Streamlit server, and you can access the application in your browser at http://localhost:8501 or try the deployed version at https://askmypdfanything.streamlit.app/

## 📖 How to Use the Application

1. **Upload PDFs** 📤
   - Use the file uploader on the sidebar to upload one or more PDF files.

2. **Process PDFs** ⚙️
   - After uploading the PDFs, click the "Process" button. The app will extract text, split it into manageable chunks, and create a vector store for fast retrieval.

3. **Ask Questions** 💭
   - Once the PDFs are processed, you can ask questions by typing them in the text box and hitting Enter. The bot will provide answers based on the content of the PDFs.

4. **Chat History** 📝
   - The application maintains a chat history, allowing you to refer back to previous interactions and ask follow-up questions.

## 📁 Project Structure

Here's the structure of the project:

```
app.py                  # Main Streamlit application file
htmlTemplates.py        # Contains HTML templates for the chat interface
.streamlit/secrets.toml # Stores the Hugging Face API token
requirements.txt        # Lists the dependencies required for the project
```

## ⚠️ Notes

- This project works best with PDFs in English, as the language model used (Flan-T5) is trained on English text.
- Be aware that text extraction may not work perfectly on PDFs with complex layouts, scanned images, or intricate formatting.
- An active internet connection is required since the app relies on Hugging Face's Flan-T5 model to generate responses.
