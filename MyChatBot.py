import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

# Configure Gemini API (Free alternative to OpenAI)
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"  # Get free API key from https://makersuite.google.com/app/apikey

st.header("NoteBot - Free Version")

with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload notes PDF and start asking questions", type="pdf")

    # Add API key input
    api_key_input = st.text_input("Enter your Google AI API Key (optional)", type="password",
                                  help="Get free API key from https://makersuite.google.com/app/apikey")


# Initialize embeddings model (free)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )


# extracting the text from pdf file
if file is not None:
    try:
        my_pdf = PdfReader(file)
        text = ""
        for page in my_pdf.pages:
            text += page.extract_text()

        if not text.strip():
            st.warning("No text found in the PDF. Please make sure the PDF contains readable text.")
        else:
            # break it into Chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=300, chunk_overlap=50, length_function=len)
            chunks = splitter.split_text(text)

            st.success(f"PDF processed! Found {len(chunks)} text chunks.")

            # Load embeddings model
            with st.spinner("Loading embeddings model..."):
                embeddings = load_embeddings()

            # Creating VectorDB & Storing embeddings into it
            with st.spinner("Creating vector store..."):
                vector_store = FAISS.from_texts(chunks, embeddings)

            # get user query
            user_query = st.text_input("Type your query here")

            # Process query if provided
            if user_query and user_query.strip():
                # semantic search from vector store
                matching_chunks = vector_store.similarity_search(user_query, k=3)

                # Check if we have API key for advanced responses
                current_api_key = api_key_input if api_key_input else GOOGLE_API_KEY

                if current_api_key and current_api_key != "YOUR_GOOGLE_API_KEY_HERE":
                    try:
                        # Configure Gemini
                        genai.configure(api_key=current_api_key)

                        # Initialize Gemini model (free tier available)
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-pro",
                            temperature=0,
                            google_api_key=current_api_key
                        )

                        # Create prompt template
                        customized_prompt = ChatPromptTemplate.from_template(
                            """You are my assistant tutor. Answer the question based on the following context.
                            If you cannot find the answer in the context, simply say "I don't know Jenny".

                            Context: {context}
                            Question: {input}

                            Answer:"""
                        )

                        # Create chain and get response
                        chain = create_stuff_documents_chain(llm, customized_prompt)

                        # Prepare context from matching chunks
                        context_text = "\n\n".join([doc.page_content for doc in matching_chunks])

                        with st.spinner("Generating response..."):
                            response = chain.invoke({
                                "input": user_query,
                                "context": context_text
                            })

                        st.write("**Answer:**")
                        st.write(response)

                    except Exception as e:
                        st.error(f"Error with AI model: {str(e)}")
                        st.write("**Fallback - Relevant text chunks:**")
                        for i, chunk in enumerate(matching_chunks, 1):
                            st.write(f"**Chunk {i}:**")
                            st.write(chunk.page_content)
                            st.write("---")
                else:
                    st.info(
                        "💡 Add your Google AI API key for AI-powered responses, or see the relevant text chunks below:")
                    st.write("**Relevant text chunks from your notes:**")
                    for i, chunk in enumerate(matching_chunks, 1):
                        st.write(f"**Chunk {i}:**")
                        st.write(chunk.page_content)
                        st.write("---")

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")

# Instructions
st.sidebar.markdown("---")
st.sidebar.markdown("### Instructions:")
st.sidebar.markdown("1. Upload a PDF file")
st.sidebar.markdown("2. Get a free Google AI API key from [here](https://makersuite.google.com/app/apikey)")
st.sidebar.markdown("3. Enter your API key above")
st.sidebar.markdown("4. Ask questions about your PDF!")

st.sidebar.markdown("---")
st.sidebar.markdown("### Free Features:")
st.sidebar.markdown("✅ PDF text extraction")
st.sidebar.markdown("✅ Free embeddings")
st.sidebar.markdown("✅ Semantic search")
st.sidebar.markdown("✅ Google Gemini AI (free tier)")

if st.sidebar.button("Clear Cache"):
    st.cache_resource.clear()
    st.success("Cache cleared!")