import os
import logging
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntranetRepository:
    _instance = None  # Singleton instance
    _vectorstore = None 
    
    # Configuration constants
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(IntranetRepository, cls).__new__(cls)
        return cls._instance

    def __init__(self, index_path="faiss_index"):
        self.index_path = index_path
        self.docs_file = os.path.join("docs", "Delta_Logistic_Intranet.txt")
        
    def load_local_document(self):
        """Load the local document file."""
        try:
            if not os.path.exists(self.docs_file):
                logger.error(f"Document file not found: {self.docs_file}")
                return None
                
            logger.info(f"Loading document from {self.docs_file}")
            
            # Read file content with encoding handling
            try:
                with open(self.docs_file, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decoding failed, trying latin-1")
                with open(self.docs_file, 'r', encoding='latin-1') as f:
                    text_content = f.read()
            
            # Create a single document
            doc = Document(
                page_content=text_content,
                metadata={'source': os.path.basename(self.docs_file)}
            )
            
            # Split into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.CHUNK_SIZE,
                chunk_overlap=self.CHUNK_OVERLAP
            )
            
            chunks = text_splitter.split_documents([doc])
            
            # Ensure all chunks maintain source information
            for chunk in chunks:
                chunk.metadata['source'] = os.path.basename(self.docs_file)
                
            logger.info(f"Split document into {len(chunks)} chunks of ~{self.CHUNK_SIZE} chars")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return []

    def create_or_load_faiss_index(self, force_rebuild=False):
        """
        Create or load a FAISS index.
        If force_rebuild is True, it will rebuild the index even if it exists.
        """
        # If we have the vectorstore in memory and don't need to rebuild, return it
        if IntranetRepository._vectorstore is not None and not force_rebuild:
            logger.info("Reusing existing FAISS index from memory.")
            return IntranetRepository._vectorstore

        # If the index exists on disk and we don't need to rebuild, load it
        if os.path.exists(self.index_path) and os.path.isfile(f"{self.index_path}/index.faiss") and not force_rebuild:
            logger.info(f"Loading FAISS index from {self.index_path}")
            try:
                IntranetRepository._vectorstore = FAISS.load_local(
                    self.index_path,
                    OpenAIEmbeddings(),
                    allow_dangerous_deserialization=True
                )
                logger.info("Successfully loaded FAISS index")
                return IntranetRepository._vectorstore
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                logger.info("Will rebuild the index...")
                # Continue to rebuild the index
        
        # Create new index
        logger.info(f"Creating FAISS index from local document")
        
        try:
            # Load and process document
            chunks = self.load_local_document()
            
            if not chunks:
                logger.warning("No chunks created from the document.")
                return None
            
            # Create embeddings and FAISS index
            logger.info(f"Creating FAISS index from {len(chunks)} chunks")
            embeddings = OpenAIEmbeddings()
            
            # Create index in batches to avoid memory issues
            batch_size = 100
            if len(chunks) <= batch_size:
                # Create index directly if number of chunks is small
                IntranetRepository._vectorstore = FAISS.from_documents(chunks, embeddings)
            else:
                # Create index in batches for larger sets
                logger.info(f"Creating index in batches of {batch_size} chunks")
                # Create first batch
                first_batch = chunks[:batch_size]
                IntranetRepository._vectorstore = FAISS.from_documents(first_batch, embeddings)
                
                # Add remaining batches
                for i in range(batch_size, len(chunks), batch_size):
                    end_idx = min(i + batch_size, len(chunks))
                    logger.info(f"Adding batch {i//batch_size + 1}: chunks {i} to {end_idx}")
                    batch = chunks[i:end_idx]
                    if batch:
                        IntranetRepository._vectorstore.add_documents(batch)
            
            # Ensure the directory exists
            os.makedirs(self.index_path, exist_ok=True)
            
            # Save the index
            IntranetRepository._vectorstore.save_local(self.index_path)
            logger.info(f"FAISS index created and saved to {self.index_path}")
            
            return IntranetRepository._vectorstore
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            return None

    def query_document(self, question, k=3):
        """Query the FAISS index with a question and return relevant context."""
        if IntranetRepository._vectorstore is None:
            logger.error("FAISS index is not loaded. Trying to load it now.")
            self.create_or_load_faiss_index()
            
            if IntranetRepository._vectorstore is None:
                raise ValueError("Failed to load FAISS index. Call create_or_load_faiss_index first.")
        
        docs = IntranetRepository._vectorstore.similarity_search(question, k=k)
        
        if docs:
            # Format the results to include source information
            results = []
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content
                results.append(f"[Source: {source}]\n{content}")
            
            return "\n\n".join(results)
        
        return "No relevant information found."
        
    def force_rebuild_index(self):
        """Force rebuild the index from scratch."""
        logger.info(f"Attempting to remove existing index at {self.index_path}")
        
        try:
            # Check and remove specific files first
            index_files = [
                os.path.join(self.index_path, "index.faiss"),
                os.path.join(self.index_path, "index.pkl")
            ]
            
            for file_path in index_files:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"Successfully removed {file_path}")
                    except Exception as e:
                        logger.error(f"Error removing {file_path}: {e}")
            
            # Check if directory is empty and remove it if it is
            if os.path.exists(self.index_path) and os.path.isdir(self.index_path):
                if not os.listdir(self.index_path):  # If empty
                    try:
                        os.rmdir(self.index_path)
                        logger.info(f"Successfully removed empty directory {self.index_path}")
                    except Exception as e:
                        logger.error(f"Error removing directory {self.index_path}: {e}")
        except Exception as e:
            logger.error(f"Error during index cleanup: {e}")
        
        # Reset vectorstore - IMPORTANT: reset class variable
        logger.info("Resetting vector store in memory")
        IntranetRepository._vectorstore = None
        
        # Recreate index with force_rebuild=True to ensure new creation
        logger.info("Rebuilding index from scratch")
        return self.create_or_load_faiss_index(force_rebuild=True)