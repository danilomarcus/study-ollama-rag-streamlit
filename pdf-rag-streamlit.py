import streamlit as st
import os
import logging
import warnings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama
import pyperclip  # Para copiar para a área de transferência

# Configure logging
logging.basicConfig(level=logging.INFO)

# constants
PDF_PATH = '/home/danilo/Develops/ai/dados.pdf'
EMBEDDING_MODEL = "mxbai-embed-large"
RAG_MODEL = "llama3.2"
COLLECTION_NAME = "software-analyst-rag"
PERSIST_DIRECTORY = "./chroma_db"

def load_pdf(doc_path):
    """ Carrega o documento PDF """
    if not os.path.exists(doc_path):
        logging.error(f"PDF file not found at path: {doc_path}")
        st.error(f"PDF file not found at path: {doc_path}")
        return None
    
    try:
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF successfully loaded!")
        return data
    except Exception as e:
        logging.error(f"Error loading PDF: {e}")
        st.error(f"Error loading PDF: {e}")
        return None

def split_documents(documents):
    """Split documents into smaller chunks."""
    if not documents:
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Documents split into {len(chunks)} chunks.")
    return chunks

@st.cache_resource
def load_vector_database(_pdf_path=PDF_PATH):
    """Load or create the vector database from the persistence directory."""
    try:
        # pull the embedding model if not exists
        ollama.pull(EMBEDDING_MODEL)
        
        embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        # Verificar se o diretório de persistência existe e contém dados
        if os.path.exists(PERSIST_DIRECTORY) and len(os.listdir(PERSIST_DIRECTORY)) > 0:
            # Carregar banco de dados existente
            vector_db = Chroma(
                embedding_function=embedding,
                persist_directory=PERSIST_DIRECTORY,
                collection_name=COLLECTION_NAME,
            )
            logging.info("Existing vector database loaded")
            return vector_db
        else:
            # Load and process the PDF document
            documents = load_pdf(_pdf_path)
            if documents is None:
                return None
            
            # split documents into chunks
            chunks = split_documents(documents)
            if not chunks:
                st.error("No content was extracted from the PDF.")
                return None
                
            # create vector database
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding,
                collection_name=COLLECTION_NAME,
                persist_directory=PERSIST_DIRECTORY,
            )
            logging.info("Vector database created and persisted")
            return vector_db
    except Exception as e:
        logging.error(f"Error in vector database setup: {e}")
        st.error(f"Error in vector database setup: {e}")
        return None

def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    if vector_db is None:
        return None
        
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant specialized in software development and system analysis. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )

    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(),
        llm=llm,
        prompt=QUERY_PROMPT,
    )
    logging.info("Retriever created.")
    return retriever

def create_rag_chain(retriever, llm):
    """Create the chain"""
    if retriever is None:
        return None
        
    # RAG prompt especializado em desenvolvimento de software
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
    logging.info("RAG chain created.")
    return chain

def main():
    """Main function to execute the RAG pipeline."""
    st.title("Análise de escopo")
    
    # Seleção de arquivo
    uploaded_file = st.file_uploader("Carregue um arquivo PDF", type="pdf")
    pdf_path = PDF_PATH  # Caminho padrão
    
    # Se um arquivo foi carregado, salve-o e use esse caminho
    if uploaded_file is not None:
        with open("temp_upload.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_path = "temp_upload.pdf"
        st.success("Arquivo carregado com sucesso!")

    # Controle de temperatura do modelo
    st.sidebar.header("Configurações do Modelo")
    temperature = st.sidebar.slider("Temperatura (criatividade)", min_value=0.0, max_value=1.0, value=0.1, step=0.1, 
                                    help="Valores mais baixos (próximos a 0) tornam as respostas mais determinísticas e focadas nos fatos.")
    
    # User input
    user_input = st.text_area("Digite sua pergunta:", "")

    if "last_response" not in st.session_state:
        st.session_state.last_response = ""

    if user_input:
        with st.spinner("Processando..."):
            try:
                # initialize RAG model com temperatura baixa
                llm = ChatOllama(model=RAG_MODEL, temperature=temperature)

                # Usar o caminho do PDF carregado ou o padrão
                vector_db = load_vector_database(pdf_path)
                if vector_db is None:
                    st.error("Erro ao carregar o banco de dados vetoriais.")
                    return
                
                # Create retriever
                retriever = create_retriever(vector_db, llm)
                if retriever is None:
                    st.error("Erro ao criar o retriever.")
                    return

                # Create RAG chain
                chain = create_rag_chain(retriever, llm)
                if chain is None:
                    st.error("Erro ao criar a cadeia RAG.")
                    return

                # Get response from RAG chain
                response = chain.invoke(input=user_input)
                st.session_state.last_response = response

                # Exibir resposta com botão para copiar
                st.markdown("**Assistente:**")
                st.write(response)
                
                # Botão para copiar a resposta
                if st.button("📋 Copiar resposta para a área de transferência"):
                    try:
                        pyperclip.copy(response)
                        st.success("Resposta copiada com sucesso!")
                    except Exception as e:
                        st.error(f"Erro ao copiar: {str(e)}")
                
            except Exception as e:
                st.error(f"Erro no processamento: {str(e)}")
                logging.error(f"Error in processing: {e}")
    else:
        st.info("Por favor, digite uma pergunta para começar.")
    
    # Se houver uma resposta prévia, mostrar botão para copiar mesmo sem nova consulta
    if st.session_state.last_response and not user_input:
        st.markdown("**Última resposta:**")
        st.write(st.session_state.last_response)
        
        if st.button("📋 Copiar resposta anterior"):
            try:
                pyperclip.copy(st.session_state.last_response)
                st.success("Resposta copiada com sucesso!")
            except Exception as e:
                st.error(f"Erro ao copiar: {str(e)}")
        
        # Exibir em uma caixa de texto para facilitar cópia manual
        st.text_area("Resposta anterior para cópia", value=st.session_state.last_response, height=200)

if __name__ == "__main__":
    main()
