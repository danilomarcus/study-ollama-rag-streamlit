import streamlit as st
import os
import ollama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configurações da página
st.set_page_config(page_title="Especialista em Desenvolvimento de Software", layout="wide")

# Função para carregar documento
@st.cache_resource
def carregar_e_processar_pdf(arquivo_pdf):
    try:
        # Salvar arquivo temporário
        with open(arquivo_pdf.name, "wb") as f:
            f.write(arquivo_pdf.getbuffer())
        
        loader = UnstructuredPDFLoader(file_path=arquivo_pdf.name)
        st.info("Carregando o documento...")
        data = loader.load()
        
        # Dividir em chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(data)
        st.success(f"Documento processado em {len(chunks)} chunks")
        
        return chunks
    except Exception as e:
        st.error(f"Erro ao processar o documento: {e}")
        return None

# Função para criar a base vetorial
@st.cache_resource
def criar_base_vetorial(_chunks, embedding_model):
    try:
        vector_db = Chroma.from_documents(
            documents=_chunks,
            embedding=OllamaEmbeddings(model=embedding_model),
            collection_name="streamlit-rag"
        )
        return vector_db
    except Exception as e:
        st.error(f"Erro ao criar base vetorial: {e}")
        return None

# Função para configurar o RAG
def configurar_rag(vector_db, rag_model):
    llm = ChatOllama(model=rag_model)
    
    # Prompt para gerar consultas
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
    )
    
    # Configurar o retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(),
        llm=llm,
        prompt=QUERY_PROMPT,
    )
    
    # Template para RAG
    template = """Você é um especialista técnico em desenvolvimento PHP com CodeIgniter 3, integrações com SAP Business One e bancos de dados relacionais.

    Ao analisar escopos de mudanças em sistemas existentes, você avalia criteriosamente:
    - Viabilidade técnica da implementação no CodeIgniter 3
    - Complexidade de desenvolvimento e estimativa de esforço
    - Possíveis impactos em funcionalidades existentes
    - Desafios de integração com SAP Business One
    - Otimizações necessárias em consultas SQL
    - Pontos de atenção em segurança e escalabilidade

    Responda à pergunta baseando-se APENAS no seguinte contexto:
    
    {context}
    
    Se o contexto não contiver informações suficientes, indique quais documentações técnicas adicionais seriam necessárias para uma análise completa. Quando encontrar código PHP ou consultas SQL, avalie-os considerando as melhores práticas para CodeIgniter 3 e bancos de dados relacionais.
    
    Sua resposta deve ser estruturada, técnica e direta, fornecendo insights acionáveis sobre o escopo de mudanças proposto.
    
    Pergunta: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# Interface principal
st.title("Especialista em Desenvolvimento de Software")
st.subheader("Consultor de Documentação Técnica com IA")

# Sidebar
with st.sidebar:
    st.header("Configurações")
    
    # Lista de modelos disponíveis
    try:
        with st.spinner("Carregando modelos disponíveis..."):
            models = ollama.list()
            model_names = [model["name"] for model in models["models"]] if "models" in models else ["llama3.2"]
    except:
        model_names = ["llama3.2"]
    
    embedding_model = st.selectbox("Modelo de Embedding", ["mxbai-embed-large", "nomic-embed-text"], index=0)
    rag_model = st.selectbox("Modelo RAG", model_names, index=0 if "llama3.2" in model_names else 0)
    
    # Upload de arquivo
    st.header("Carregue seu documento")
    arquivo_pdf = st.file_uploader("Escolha um arquivo PDF", type=["pdf"])

    st.markdown("""
        ## Criado por:
        - [@danilomarcus](https://github.com/danilomarcus)
        """)

# Variáveis de estado da sessão
if "documento_carregado" not in st.session_state:
    st.session_state.documento_carregado = False
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "historico" not in st.session_state:
    st.session_state.historico = []

# Processar documento quando enviado
if arquivo_pdf is not None and not st.session_state.documento_carregado:
    with st.spinner("Processando documento..."):
        chunks = carregar_e_processar_pdf(arquivo_pdf)
        if chunks:
            st.session_state.vector_db = criar_base_vetorial(chunks, embedding_model)
            if st.session_state.vector_db:
                st.session_state.rag_chain = configurar_rag(st.session_state.vector_db, rag_model)
                st.session_state.documento_carregado = True
                st.success("Documento processado e base de conhecimento criada!")

# Interface de consulta
if st.session_state.documento_carregado:
    st.header("Faça sua consulta ao documento")
    
    # Campo de consulta
    query = st.text_input("Digite sua pergunta")
    col1, col2 = st.columns([1, 5])
    
    with col1:
        submit_button = st.button("Consultar", use_container_width=True)
    
    with col2:
        clear_button = st.button("Limpar histórico", use_container_width=True)
        if clear_button:
            st.session_state.historico = []
            st.rerun()
    
    # Processar consulta
    if submit_button and query:
        with st.spinner("Gerando resposta..."):
            try:
                resposta = st.session_state.rag_chain.invoke(input=query)
                st.session_state.historico.append({"pergunta": query, "resposta": resposta})
            except Exception as e:
                st.error(f"Erro ao processar consulta: {e}")
    
    # Mostrar histórico de consultas
    if st.session_state.historico:
        st.subheader("Histórico de consultas")
        for i, item in enumerate(reversed(st.session_state.historico)):
            with st.expander(f"Consulta: {item['pergunta']}", expanded=(i == 0)):
                st.markdown(item["resposta"])
else:
    if arquivo_pdf is None:
        st.info("👈 Por favor, faça upload de um arquivo PDF na barra lateral para começar.")
    else:
        st.warning("Aguarde o processamento do documento...")
        
# Instruções
if not st.session_state.documento_carregado:
    st.subheader("Como usar:")
    st.markdown("""
    1. Selecione os modelos de embedding e RAG na barra lateral
    2. Faça upload de um documento PDF contendo especificações técnicas ou código
    3. Após o processamento, faça perguntas sobre o documento
    4. O sistema analisará o documento e responderá com insights técnicos relevantes
    """)

# Rodapé
st.sidebar.markdown("---")
st.sidebar.caption("Desenvolvido com LangChain e Ollama") 