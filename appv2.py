import streamlit as st
from chains import first_responder, final_responder, global_responder, salary_responder, vacancy_responder
from langgraph.graph import MessageGraph
from classes import FinalResponse
from services.Intranet_repository_s3 import IntranetRepository
import json
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import os
import time
from dotenv import load_dotenv
import boto3
import tempfile
import shutil

# Configurações da página Streamlit
st.set_page_config(
    page_title="Delta Logistic Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilo CSS personalizado para ocultar o botão de expansão da barra lateral
hide_sidebar_button = """
<style>
    [data-testid="collapsedControl"] {
        display: none;
    }
</style>
"""
st.markdown(hide_sidebar_button, unsafe_allow_html=True)

# Função de configuração AWS
def configure_aws():
    """
    Configure AWS credentials from environment variables.
    """
    load_dotenv()
    
    # Check if AWS credentials are set in environment variables
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION', 'us-east-1')
    
    if not aws_access_key or not aws_secret_key:
        raise ValueError(
            "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY environment variables."
        )
    
    # Configure boto3 session
    boto3.setup_default_session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
    
    print(f"AWS configured with region: {aws_region}")
    
    return {
        'region': aws_region,
        'credentials_found': True
    }

# Função para fazer upload de arquivo para S3
def upload_file_to_s3(bucket_name, file_object, object_name=None):
    """
    Upload a file to an S3 bucket
    
    :param bucket_name: Bucket to upload to
    :param file_object: File-like object to upload
    :param object_name: S3 object name. If not specified then the file name is used
    :return: True if file was uploaded, else False
    """
    # Se o nome do objeto não for especificado, use o nome do arquivo
    if object_name is None:
        object_name = file_object.name
        
    # Upload the file
    s3_client = boto3.client('s3')
    try:
        # Criar um arquivo temporário para salvar o conteúdo do arquivo carregado
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_object.getvalue())
            temp_file_path = temp_file.name
        
        # Upload do arquivo do disco para o S3
        s3_client.upload_file(temp_file_path, bucket_name, object_name)
        
        # Remover o arquivo temporário
        os.remove(temp_file_path)
        
        return True
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        return False

# Função para visualizar conteúdo de um arquivo S3
def view_s3_file_content(bucket_name, file_key):
    """
    Retrieve and display content of a file from S3
    """
    s3_client = boto3.client('s3')
    try:
        # Baixar arquivo do S3
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            s3_client.download_file(bucket_name, file_key, temp_file.name)
            temp_file_path = temp_file.name
        
        # Ler conteúdo do arquivo
        with open(temp_file_path, 'r') as f:
            content = f.read()
        
        # Limpar arquivo temporário
        os.remove(temp_file_path)
        
        return content
    except Exception as e:
        print(f"Error retrieving file from S3: {e}")
        return f"Erro ao recuperar arquivo: {str(e)}"

# Função de diagnóstico de índice FAISS
def diagnose_faiss_index(repository):
    """
    Diagnostic function to check FAISS index details
    """
    st.sidebar.subheader("Diagnóstico do Índice")
    
    with st.sidebar.expander("Ver Diagnóstico"):
        if repository._vectorstore is None:
            st.warning("Índice FAISS não está carregado na memória.")
            return
        
        # Verificar quantos documentos estão indexados
        try:
            # Método direto para ver número de vetores (pode variar dependendo da implementação exata)
            if hasattr(repository._vectorstore, 'index'):
                num_vectors = repository._vectorstore.index.ntotal
                st.info(f"Número de vetores no índice: {num_vectors}")
            elif hasattr(repository._vectorstore, 'docstore'):
                num_docs = len(repository._vectorstore.docstore._dict)
                st.info(f"Número de documentos no índice: {num_docs}")
            else:
                st.warning("Não foi possível determinar o tamanho do índice.")
            
            # Listar metadados de documentos
            if hasattr(repository._vectorstore, 'docstore') and hasattr(repository._vectorstore.docstore, '_dict'):
                st.subheader("Documentos Indexados:")
                docs_list = list(repository._vectorstore.docstore._dict.values())
                
                # Agrupe por fonte
                source_counts = {}
                for doc in docs_list:
                    source = doc.metadata.get('source', 'Unknown')
                    if source in source_counts:
                        source_counts[source] += 1
                    else:
                        source_counts[source] = 1
                
                # Mostrar contagem por fonte
                for source, count in source_counts.items():
                    st.write(f"- {source}: {count} chunks")
                
                # Mostrar amostra de documentos
                st.subheader("Amostra de documentos:")
                for i, doc in enumerate(docs_list[:3]):  # Apenas os 3 primeiros
                    st.markdown(f"**Documento {i+1}:**")
                    st.markdown(f"**Fonte:** {doc.metadata.get('source', 'Unknown')}")
                    st.markdown(f"**Conteúdo:** {doc.page_content[:200]}...")
        except Exception as e:
            st.error(f"Erro ao analisar índice: {str(e)}")

# Função de reindexação forçada completa
# Função de reindexação forçada completa
def force_full_reindex(repository, bucket_name):
    """
    Force complete reindexing of documents from S3 bucket with detailed steps
    """
    st.sidebar.subheader("Reindexação Forçada")
    
    if st.sidebar.button("Forçar Reindexação Completa"):
        with st.sidebar:
            with st.spinner("Reindexando documentos..."):
                try:
                    st.info("1. Removendo índice existente...")
                    
                    # Método mais seguro e explícito para remover os arquivos do índice FAISS
                    index_path = "faiss_index"
                    index_files = [
                        os.path.join(index_path, "index.faiss"),
                        os.path.join(index_path, "index.pkl")
                    ]
                    
                    # Verificar e remover cada arquivo individualmente
                    for file_path in index_files:
                        if os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                                st.success(f"Arquivo {file_path} removido com sucesso.")
                            except Exception as e:
                                st.error(f"Erro ao remover {file_path}: {str(e)}")
                    
                    # Verificar se o diretório existe e está vazio e então removê-lo
                    if os.path.exists(index_path) and os.path.isdir(index_path):
                        if not os.listdir(index_path):  # Se o diretório estiver vazio
                            try:
                                os.rmdir(index_path)
                                st.success(f"Diretório {index_path} removido com sucesso.")
                            except Exception as e:
                                st.error(f"Erro ao remover diretório {index_path}: {str(e)}")
                    
                    # Reset the singleton instance
                    st.info("2. Reiniciando instância do repositório...")
                    repository._vectorstore = None
                    
                    # Limpar a referência global
                    if "repository" in st.session_state:
                        st.session_state.repository = None
                    
                    # Listar documentos no bucket
                    s3_client = boto3.client('s3')
                    response = s3_client.list_objects_v2(Bucket=bucket_name)
                    
                    if 'Contents' in response:
                        doc_count = len(response['Contents'])
                        st.info(f"3. Encontrados {doc_count} documentos no bucket S3.")
                        
                        # Listar documentos
                        for item in response['Contents']:
                            st.write(f"- {item['Key']} ({item['Size']} bytes)")
                    else:
                        st.warning("Nenhum documento encontrado no bucket S3.")
                    
                    # Recreate the index with a new repository instance
                    st.info("4. Criando novo índice FAISS...")
                    start_time = time.time()
                    
                    # Criar nova instância do repositório
                    new_repository = IntranetRepository(bucket_name=bucket_name)
                    
                    # Forçar reconstrução do índice
                    vectorstore = new_repository.force_rebuild_index()
                    elapsed_time = time.time() - start_time
                    
                    # Atualizar referência na sessão
                    st.session_state.repository = new_repository
                    
                    # Verificar se o índice foi criado
                    if vectorstore is not None:
                        st.success(f"Índice FAISS criado com sucesso em {elapsed_time:.2f} segundos.")
                        
                        # Verificar se o índice foi salvo corretamente
                        if os.path.exists(os.path.join(index_path, "index.faiss")) and \
                           os.path.exists(os.path.join(index_path, "index.pkl")):
                            st.success("Arquivos do índice verificados e estão presentes no disco.")
                        else:
                            st.warning("Os arquivos do índice podem não ter sido salvos corretamente.")
                    else:
                        st.error("Falha ao criar índice FAISS.")
                except Exception as e:
                    st.error(f"Erro ao reindexar documentos: {str(e)}")
                    st.error(f"Detalhes: {type(e).__name__}")

# Função de upload de documentos
def upload_document_section(bucket_name):
    """
    Cria uma seção para upload de documentos para o bucket S3
    """
    st.sidebar.subheader("Upload de Documentos")
    with st.sidebar.expander("Enviar Novo Documento"):
        uploaded_file = st.file_uploader("Escolha um arquivo", type=['txt', 'md', 'csv', 'json'])
        
        custom_filename = st.text_input("Nome personalizado (opcional)", 
                                        help="Se não especificado, será usado o nome original do arquivo")
        
        if st.button("Enviar Documento"):
            if uploaded_file is not None:
                with st.spinner("Enviando documento para S3..."):
                    # Definir nome do objeto S3
                    object_name = custom_filename.strip() if custom_filename.strip() else uploaded_file.name
                    
                    # Upload para S3
                    success = upload_file_to_s3(bucket_name, uploaded_file, object_name)
                    
                    if success:
                        st.success(f"Documento '{object_name}' enviado com sucesso!")
                        st.info("Por favor, reindexe os documentos para incluir o novo arquivo.")
                    else:
                        st.error("Erro ao enviar o documento. Verifique os logs para mais detalhes.")
            else:
                st.warning("Por favor, selecione um arquivo para enviar.")

# Função para explorar documentos do bucket
def explore_s3_documents(bucket_name):
    """
    Explorar e visualizar o conteúdo de documentos armazenados no bucket S3
    """
    st.sidebar.subheader("Explorar Documentos")
    
    with st.sidebar.expander("Ver Documentos do S3"):
        try:
            # Listar documentos no bucket
            s3_client = boto3.client('s3')
            response = s3_client.list_objects_v2(Bucket=bucket_name)
            
            if 'Contents' in response:
                # Criar seletor de documento
                doc_options = [item['Key'] for item in response['Contents']]
                selected_doc = st.selectbox("Selecione um documento para visualizar:", doc_options)
                
                if selected_doc:
                    st.write(f"**Documento:** {selected_doc}")
                    
                    # Botão para visualizar conteúdo
                    if st.button("Visualizar Conteúdo"):
                        with st.spinner("Carregando conteúdo..."):
                            content = view_s3_file_content(bucket_name, selected_doc)
                            st.text_area("Conteúdo do Documento:", value=content, height=300)
            else:
                st.warning("Nenhum documento encontrado no bucket.")
        except Exception as e:
            st.error(f"Erro ao listar documentos: {str(e)}")

# Verificação de autenticação para a barra lateral
def check_password():
    """
    Verifica se a senha está correta.
    Retorna True se a senha for correta ou já estiver autenticado.
    """
    # Inicializar estado de autenticação
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    # Se já estiver autenticado, retorna True
    if st.session_state.authenticated:
        return True
    
    # Caso contrário, mostra o formulário de senha
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h4 style='text-align: center;'>Área Administrativa</h4>", unsafe_allow_html=True)
        password = st.text_input("Digite a senha para acessar a barra lateral:", type="password")
        
        if password:
            if password == "DO2025":
                st.session_state.authenticated = True
                st.success("Senha correta! Carregando área administrativa...")
                st.rerun()
                return True
            else:
                st.error("Senha incorreta!")
                return False
        else:
            return False

# Botão para abrir a área administrativa
def show_admin_button():
    col1, col2, col3 = st.columns([4, 1, 4])
    with col2:
        if st.button("Área Admin"):
            st.session_state.show_login = True
            st.rerun()

def create_graph():
    builder = MessageGraph()
    builder.add_node("classifier", first_responder)
    builder.add_node("global", global_responder)
    builder.add_node("salary", salary_responder)
    builder.add_node("vacancy", vacancy_responder)
    builder.add_node("final", final_responder)
    builder.add_conditional_edges("classifier", decision_flow)
    builder.add_edge("global", "final")
    builder.add_edge("salary", "final")
    builder.add_edge("vacancy", "final")
    builder.set_entry_point("classifier")
    return builder.compile()

def decision_flow(state: list[BaseMessage]) -> str:
    last_message = state[-1]
    if hasattr(last_message, 'additional_kwargs') and 'tool_calls' in last_message.additional_kwargs:
        tool_calls = last_message.additional_kwargs['tool_calls']
        if tool_calls:
            last_tool = tool_calls[-1]
            arguments = last_tool['function']['arguments']
            result = json.loads(arguments)
            if result.get("request_type") == "global_question":
                return "global"
            elif result.get("request_type") == "salary_request":
                return "salary"
            elif result.get("request_type") == "vacancy_request":
                return "vacancy"
    return "final"

# Inicializar variáveis de estado
if "show_login" not in st.session_state:
    st.session_state.show_login = False
if "history" not in st.session_state:
    st.session_state.history = []

# Carregar grafo diretamente (sem usar cache_resource para evitar problemas)
if "graph" not in st.session_state:
    st.session_state.graph = create_graph()

# Configurar AWS (simplificado para evitar erros)
try:
    aws_config = configure_aws()
except Exception as e:
    if st.session_state.get("authenticated", False):
        st.sidebar.error(f"Erro ao configurar AWS: {str(e)}")

# Configuração do repositório S3
BUCKET_NAME = "docs-intranet"
try:
    repository = IntranetRepository(bucket_name=BUCKET_NAME)
    st.session_state.repository = repository
    # Carregar vectorstore
    vectorstore = repository.create_or_load_faiss_index()
except Exception as e:
    if st.session_state.get("authenticated", False):
        st.sidebar.error(f"Erro ao carregar repositório: {str(e)}")
    repository = None
    vectorstore = None

# Área de chat limpa (apenas o chat, sem outros elementos)
# Alterando o título para ficar mais sutil na parte superior
st.markdown("<h3 style='text-align: center;'>Delta Logistic Assistant</h3>", unsafe_allow_html=True)

# Se a senha estiver correta, mostrar a barra lateral
if not st.session_state.show_login:
    # Mostrar o botão para abrir a área admin
    show_admin_button()
elif not check_password():
    # A senha está sendo verificada, não mostrar o chat ainda
    pass
else:
    # Autenticado, mostrar a barra lateral com todas as informações
    with st.sidebar:
        st.title("Delta Logistic Admin")
        
        # Exibir status da AWS
        try:
            aws_config = configure_aws()
            st.success(f"AWS configurado com a região: {aws_config['region']}")
        except Exception as e:
            st.error(f"Erro ao configurar AWS: {str(e)}")
        
        if repository:
            # Mostrar documentos disponíveis
            st.subheader("Documentos no S3")
            with st.expander("Ver documentos disponíveis"):
                documents = repository.list_documents_in_bucket()
                if documents:
                    for doc in documents:
                        st.write(f"- {doc}")
                else:
                    st.write("Nenhum documento encontrado no bucket.")
            
            # Adicionar funcionalidade de exploração
            explore_s3_documents(BUCKET_NAME)
            
            # Adicionar funcionalidade de upload
            upload_document_section(BUCKET_NAME)
            
            # Adicionar diagnóstico do índice
            diagnose_faiss_index(repository)
            
            # Adicionar reindexação forçada
            force_full_reindex(repository, BUCKET_NAME)
        
        # Botão para sair da área admin
        if st.button("Sair da Área Admin"):
            st.session_state.authenticated = False
            st.session_state.show_login = False
            st.rerun()

# Se não estiver na tela de login, mostrar o chat normal
if not st.session_state.show_login or st.session_state.authenticated:
    # Exibir histórico de conversa
    for message in st.session_state.history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # Interface de entrada de chat
    query = st.chat_input("Pergunte algo...")
    if query:
        st.session_state.history.append(HumanMessage(content=query))
        MAX_HISTORY = 10
        st.session_state.history = st.session_state.history[-MAX_HISTORY:]
        try:
            # Usando container vazio para evitar mostrar o spinner na área de chat
            with st.empty():
                # Usar o grafo da session_state para evitar problemas
                response = st.session_state.graph.invoke(st.session_state.history)

                final_result_json = response[-1].content
                final_result_pydantic = FinalResponse.model_validate_json(final_result_json)
                answer = final_result_pydantic.answer

                st.session_state.history.append(AIMessage(content=answer))
                
                # Recarregar a página para mostrar a nova mensagem
                st.rerun()
        except Exception as e:
            st.session_state.history.append(AIMessage(content=f"Ocorreu um erro: {str(e)}"))
            st.rerun()