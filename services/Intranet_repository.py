import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class IntranetRepository:
    _instance = None  # Singleton instance
    _vectorstore = None 

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(IntranetRepository, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, file_path="docs/Delta_Logistic_Intranet.txt", 
                 index_path="faiss_index"):
        self.file_path = file_path
        self.index_path = index_path

    def create_or_load_faiss_index(self):
        if IntranetRepository._vectorstore is not None:
            print("Reusing existing FAISS index from memory.")
            return IntranetRepository._vectorstore

        if os.path.exists(self.index_path) and os.path.isfile(f"{self.index_path}/index.faiss"):
            print(f"Loading FAISS index from {self.index_path}")
            IntranetRepository._vectorstore = FAISS.load_local(
                self.index_path,
                OpenAIEmbeddings(),
                allow_dangerous_deserialization=True
            )
        else:
            print(f"Creating FAISS index from {self.file_path}")
            loader = TextLoader(self.file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings()
            IntranetRepository._vectorstore = FAISS.from_documents(chunks, embeddings)
            IntranetRepository._vectorstore.save_local(self.index_path)
            print(f"FAISS index created and saved to {self.index_path}")

        return IntranetRepository._vectorstore

    def query_document(self, question, k=3):
        if IntranetRepository._vectorstore is None:
            raise ValueError("FAISS index is not loaded. Call create_or_load_faiss_index first.")
        docs = IntranetRepository._vectorstore.similarity_search(question, k=k)
        if docs:
            return "\n".join([doc.page_content for doc in docs])
        return "No relevant information found."
