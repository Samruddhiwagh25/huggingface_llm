import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain_community.callbacks import get_openai_callback
from langchain import HuggingFaceHub


# Function to process PDF and return text chunks
def process_pdf(pdf_file, max_pages=50):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    cnt = 0
    for page in pdf_reader.pages:
        text += page.extract_text()
        cnt += 1
        if cnt >= max_pages:
            break

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    return chunks

# Function to load model and run QA
def run_qa(chunks, query):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    docs = VectorStore.similarity_search(query=query, k=3)
    llm = HuggingFaceHub(huggingfacehub_api_token='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',
                          repo_id="google/flan-t5-xxl",
                          model_kwargs={"temperature": 0.7, "max_length":4096}) 
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query, max_length=4096)  
    return response

# Main Streamlit app
def main():
    st.title("PDF Question Answering Chatbot")

    # Define placeholder for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        st.markdown("---")

        # Process PDF when uploaded
        chunks = process_pdf(uploaded_file)

        # Display chat history
        st.markdown("<h3 style='color: #008080;'>Chat History:</h3>", unsafe_allow_html=True)
        for idx, (query, response) in enumerate(st.session_state.chat_history):
            st.markdown(f"<p style='color: #000080;'>User {idx + 1}: {query}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #800000;'>Chatbot {idx + 1}: {response}</p>", unsafe_allow_html=True)
            st.markdown("---")

        # Ask a question
        user_input = st.text_area("Ask a question", key="user_input")

        if st.button("Ask"):
            question = st.session_state["user_input"]
            response = run_qa(chunks, question)
            st.session_state.chat_history.append((question, response))

            # Display the response
            st.markdown("<h3 style='color: #008080;'>Chatbot Response:</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #800000;'>{response}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
