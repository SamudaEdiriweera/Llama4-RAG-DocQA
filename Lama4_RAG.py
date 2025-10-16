""" Building the Llama 4 RAG Pipeline """
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


### 1. Initiating the LLM and the embedding

# --> set up the model object using the Groq API, providing it with the model name and API key
# --> download the embedding model from Hugging Face and load it as our embedding model

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=os.getenv("GROQ_API_KEY"), temperature=0)
embed_model = HuggingFaceEmbeddings(model_name = "mixedbread-ai/mxbai-embed-large-v1")

### 2. Loading and splitting the data

from langchain_community.document_loaders import DirectoryLoader # 1. load all files from a directory
from langchain.text_splitter import RecursiveCharacterTextSplitter # 2. split the documents into smaller chunks

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n"]
)

# Load the .docx files
loader = DirectoryLoader("./", glob="*.doc", use_multithreading=True)
documents = loader.load()

# Split the documents into chunks
chunks = text_splitter.split_documents(documents)

print(len(chunks))  # Check the number of chunks created

### 3.Building and populating the vector store

# --> Initialize the Chroma vector store and provide the text chunks to the vector database
# --> The text will be converted to embeddings before being stored

from langchain_chroma import Chroma

vectore_store = Chroma.from_documents(
    documents=chunks,
    embedding=embed_model,
    # collection_name="lama4_rag",
    persist_directory="./Vectordb"
)

query = "What is this document about what it is saying?"
docs = vectore_store.similarity_search(query)
print(docs[0].page_content)

### 4.Creating the RAG pipeline
''' Next, we will convert our vector store into a retriever and create a prompt template for the RAG pipeline.'''

# Create retriever
retriever = vectore_store.as_retriever()

# Import PromptTemplate
from langchain_core.prompts import PromptTemplate

# Define a clearer, more professional prompt template
template = """
        You are an expert assistant tasked with answering questions based on the provided documents.
        Use only the given context to generate your answer.
        If the answer cannot be found in the context, clearly state that you do not know.
        Be detailed and precise in your response, but avoid mentioning or referencing the context itself.

        Context:
        {context}

        Question:
        {question}

        Answer:
    """
    
# Create the PromptTemplate object
rag_prompt = PromptTemplate.from_template(template)

# --> create the RAG chain that provides both the context and the question in the RAG prompt
# --> pass it through the Llama 4 model, and then generate a clean response

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    { "context": retriever, "question": RunnablePassthrough() }
    | rag_prompt
    | llm
    | StrOutputParser()
)

from IPython.display import display, Markdown

response = rag_chain.invoke("What is this document about what it is saying?")
Markdown(response)
print(response)