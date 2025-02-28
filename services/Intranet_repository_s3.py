import os
import tempfile
import boto3
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging
import shutil
import concurrent.futures

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntranetRepositoryS3:
    _instance = None  # Singleton instance
    _vectorstore = None 
    
    # Constantes para configuração
    CHUNK_SIZE = 300  # Tamanho reduzido dos chunks para 500 caracteres
    CHUNK_OVERLAP = 50  # Overlap menor para acompanhar o tamanho menor do chunk
    MAX_WORKERS = 4  # Número máximo de workers para processamento paralelo

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(IntranetRepositoryS3, cls).__new__(cls)
        return cls._instance

    def __init__(self, 
                 bucket_name="docs-intranet", 
                 index_path="faiss_index"):
        self.bucket_name = bucket_name
        self.index_path = index_path
        self.s3_client = boto3.client('s3')
        
    def list_documents_in_bucket(self):
        """List all documents in the S3 bucket."""
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            if 'Contents' in response:
                # Filtrar apenas arquivos relevantes (.txt, .md, .csv, .json)
                valid_extensions = ('.txt', '.md', '.csv', '.json')
                valid_files = [
                    item['Key'] for item in response['Contents']
                    if any(item['Key'].lower().endswith(ext) for ext in valid_extensions)
                ]
                
                logger.info(f"Found {len(valid_files)} valid documents in bucket {self.bucket_name}")
                return valid_files
                
            logger.warning(f"No documents found in bucket {self.bucket_name}")
            return []
        except Exception as e:
            logger.error(f"Error listing objects in S3 bucket: {e}")
            return []

    def download_file(self, file_key, temp_dir):
        """
        Download a single file from S3.
        
        Args:
            file_key: The key of the file in S3 bucket
            temp_dir: Temporary directory to save the file
            
        Returns:
            tuple: (file_key, local_file_path) or None if failed
        """
        try:
            temp_file_path = os.path.join(temp_dir, os.path.basename(file_key))
            self.s3_client.download_file(self.bucket_name, file_key, temp_file_path)
            return (file_key, temp_file_path)
        except Exception as e:
            logger.error(f"Error downloading {file_key}: {e}")
            return None

    def download_files_from_s3(self):
        """Download all files from S3 to a temporary directory and return their paths."""
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        try:
            files = self.list_documents_in_bucket()
            if not files:
                return temp_dir, []
                
            logger.info(f"Downloading {len(files)} files in parallel")
            
            # Usar processamento paralelo para download
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                futures = [executor.submit(self.download_file, file_key, temp_dir) for file_key in files]
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        file_paths.append(result)
                
            logger.info(f"Successfully downloaded {len(file_paths)} files to {temp_dir}")
            return temp_dir, file_paths
        except Exception as e:
            logger.error(f"Error in download process: {e}")
            raise

    def process_single_file(self, file_info):
        """
        Process a single file and return its documents.
        
        Args:
            file_info: Tuple of (file_key, file_path)
            
        Returns:
            list: List of Document objects or empty list if failed
        """
        file_key, file_path = file_info
        documents = []
        
        try:
            # Verificar se é um arquivo de texto
            if not any(file_path.endswith(ext) for ext in ('.txt', '.md', '.csv', '.json')):
                logger.warning(f"Skipping unsupported file type: {file_key}")
                return []
                
            # Carregar conteúdo do arquivo com tratamento de encoding
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decoding failed for {file_key}, trying latin-1")
                with open(file_path, 'r', encoding='latin-1') as f:
                    text_content = f.read()
            
            # Criar documento único
            doc = Document(
                page_content=text_content,
                metadata={'source': file_key}
            )
            
            # Dividir em chunks menores
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.CHUNK_SIZE,
                chunk_overlap=self.CHUNK_OVERLAP
            )
            
            chunks = text_splitter.split_documents([doc])
            
            # Garantir que todos os chunks mantenham a informação de origem
            for chunk in chunks:
                chunk.metadata['source'] = file_key
                
            logger.info(f"Split {file_key} into {len(chunks)} chunks of ~{self.CHUNK_SIZE} chars")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing file {file_key}: {e}")
            return []

    def load_documents_from_file_paths(self, file_paths):
        """Load documents from file paths and return a list of Document objects."""
        if not file_paths:
            logger.warning("No file paths provided to load documents from")
            return []
            
        logger.info(f"Processing {len(file_paths)} files in parallel")
        all_documents = []
        
        # Processar arquivos em paralelo
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = [executor.submit(self.process_single_file, file_info) for file_info in file_paths]
            
            for future in concurrent.futures.as_completed(futures):
                chunks = future.result()
                all_documents.extend(chunks)
        
        # Agrupar por fonte para logging
        sources = {}
        for doc in all_documents:
            source = doc.metadata.get('source', 'Unknown')
            if source in sources:
                sources[source] += 1
            else:
                sources[source] = 1
                
        # Log estatísticas
        logger.info(f"Total chunks created: {len(all_documents)}")
        for source, count in sources.items():
            logger.info(f"  {source}: {count} chunks")
            
        return all_documents

    def create_or_load_faiss_index(self, force_rebuild=False):
        """
        Create or load a FAISS index.
        If force_rebuild is True, it will rebuild the index even if it exists.
        """
        # Se temos o vectorstore em memória e não precisamos reconstruir, retorne-o
        if IntranetRepositoryS3._vectorstore is not None and not force_rebuild:
            logger.info("Reusing existing FAISS index from memory.")
            return IntranetRepositoryS3._vectorstore

        # Se o índice existir em disco e não precisamos reconstruir, carregue-o
        if os.path.exists(self.index_path) and os.path.isfile(f"{self.index_path}/index.faiss") and not force_rebuild:
            logger.info(f"Loading FAISS index from {self.index_path}")
            try:
                IntranetRepositoryS3._vectorstore = FAISS.load_local(
                    self.index_path,
                    OpenAIEmbeddings(),
                    allow_dangerous_deserialization=True
                )
                logger.info("Successfully loaded FAISS index")
                return IntranetRepositoryS3._vectorstore
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                logger.info("Will rebuild the index...")
                # Continue to rebuild the index
        
        # Create new index
        logger.info(f"Creating FAISS index from S3 documents")
        
        try:
            # Download all files from S3
            start_time = logger.info("Starting download from S3")
            temp_dir, file_paths = self.download_files_from_s3()
            
            if not file_paths:
                logger.warning("No files found in the S3 bucket.")
                return None
            
            # Load and process documents from downloaded files
            logger.info("Processing downloaded files")
            chunks = self.load_documents_from_file_paths(file_paths)
            
            if not chunks:
                logger.warning("No chunks created from files.")
                # Clean up temporary directory
                shutil.rmtree(temp_dir)
                return None
            
            # Create embeddings and FAISS index
            logger.info(f"Creating FAISS index from {len(chunks)} chunks")
            embeddings = OpenAIEmbeddings()
            
            # Criar índice em lotes para evitar problemas de memória
            batch_size = 100  # Tamanho do lote para criação do índice
            if len(chunks) <= batch_size:
                # Criar índice diretamente se o número de chunks for pequeno
                IntranetRepositoryS3._vectorstore = FAISS.from_documents(chunks, embeddings)
            else:
                # Criar índice por lotes para conjuntos maiores
                logger.info(f"Creating index in batches of {batch_size} chunks")
                # Criar primeiro lote
                first_batch = chunks[:batch_size]
                IntranetRepositoryS3._vectorstore = FAISS.from_documents(first_batch, embeddings)
                
                # Adicionar lotes restantes
                for i in range(batch_size, len(chunks), batch_size):
                    end_idx = min(i + batch_size, len(chunks))
                    logger.info(f"Adding batch {i//batch_size + 1}: chunks {i} to {end_idx}")
                    batch = chunks[i:end_idx]
                    if batch:  # Verificar se o lote não está vazio
                        IntranetRepositoryS3._vectorstore.add_documents(batch)
            
            # Ensure the directory exists
            os.makedirs(self.index_path, exist_ok=True)
            
            # Save the index
            IntranetRepositoryS3._vectorstore.save_local(self.index_path)
            logger.info(f"FAISS index created and saved to {self.index_path}")
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            logger.info(f"Temporary directory {temp_dir} removed")
            
            return IntranetRepositoryS3._vectorstore
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return None

    def query_document(self, question, k=3):
        """Query the FAISS index with a question and return relevant context."""
        if IntranetRepositoryS3._vectorstore is None:
            logger.error("FAISS index is not loaded. Trying to load it now.")
            self.create_or_load_faiss_index()
            
            if IntranetRepositoryS3._vectorstore is None:
                raise ValueError("Failed to load FAISS index. Call create_or_load_faiss_index first.")
        
        docs = IntranetRepositoryS3._vectorstore.similarity_search(question, k=k)
        
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
        # Limpar índice existente

        print("*******FORCING******")
        logger.info(f"Attempting to remove existing index at {self.index_path}")
        
        try:
            # Verificar e remover os arquivos específicos primeiro
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
            
            # Verificar se o diretório está vazio e removê-lo se estiver
            if os.path.exists(self.index_path) and os.path.isdir(self.index_path):
                if not os.listdir(self.index_path):  # Se estiver vazio
                    try:
                        os.rmdir(self.index_path)
                        logger.info(f"Successfully removed empty directory {self.index_path}")
                    except Exception as e:
                        logger.error(f"Error removing directory {self.index_path}: {e}")
        except Exception as e:
            logger.error(f"Error during index cleanup: {e}")
        
        # Resetar o vectorstore - IMPORTANTE: resetar a variável de classe
        logger.info("Resetting vector store in memory")
        IntranetRepositoryS3._vectorstore = None
        
        # Recriar o índice com force_rebuild=True para garantir nova criação
        logger.info("Rebuilding index from scratch")
        return self.create_or_load_faiss_index(force_rebuild=True)