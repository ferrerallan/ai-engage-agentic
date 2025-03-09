import streamlit as st
from services.Intranet_repository_s3 import IntranetRepository
import os
import time

def reindex_documents():
    """
    Force reindexing of documents from S3 bucket
    """
    st.sidebar.subheader("Reindexar Documentos")
    
    if st.sidebar.button("Reindexar Documentos do S3"):
        with st.sidebar:
            with st.spinner("Reindexando documentos..."):
                try:
                    # Remove existing index if it exists
                    index_path = "faiss_index"
                    if os.path.exists(f"{index_path}/index.faiss"):
                        os.remove(f"{index_path}/index.faiss")
                        os.remove(f"{index_path}/index.pkl")
                        st.info("Índice existente removido.")
                    
                    # Reset the singleton instance
                    repository = IntranetRepository(bucket_name="docs-intranet")
                    repository._vectorstore = None
                    
                    # Recreate the index
                    start_time = time.time()
                    repository.create_or_load_faiss_index()
                    elapsed_time = time.time() - start_time
                    
                    st.success(f"Documentos reindexados com sucesso em {elapsed_time:.2f} segundos.")
                except Exception as e:
                    st.error(f"Erro ao reindexar documentos: {str(e)}")