import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)

PDF_PATH = '/home/danilo/Develops/ai/dados.pdf'
EMBEDDING_MODEL = "mxbai-embed-large"
RAG_MODEL = "llama3.2"
COLLECTION_NAME = "software-analyst-rag"

def load_pdf(doc_path):
    """ Carrega o documento PDF """
    if not os.path.exists(doc_path):
        logging.error(f"PDF file not found at path: {doc_path}")
        return None
    
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    logging.info("PDF successfully loaded!")
    return data

def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks

def create_vector_database(chunks):
    """Create a vector database from document chunks."""
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=COLLECTION_NAME,
    )
    logging.info("Vector database created.")
    return vector_db

def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
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
    try:
        # Load PDF
        documents = load_pdf(PDF_PATH)
        if documents is None:
            return
        
        # Split documents into chunks
        chunks = split_documents(documents)

        # Create vector database
        vector_db = create_vector_database(chunks)

        # initialize RAG model
        llm = ChatOllama(model=RAG_MODEL)

        # Create retriever
        retriever = create_retriever(vector_db, llm)

        # Create RAG chain
        chain = create_rag_chain(retriever, llm)

        # Consultando o RAG
        # question = "Quais são os stakeholders citados no documento?"
        # question = "Quem é o responsável pela criação do documento, existe um controle de versão?"
        # question = "Do que se trata o documento?"
        question = "Do que se trata o documento? Identifique requisitos principais e possíveis desafios técnicos."
        res = chain.invoke(input=question)
        print("\n\nResposta do RAG:")
        print("-" * 50)
        print(res)
        print("-" * 50)

    except Exception as e:
        print(f"Error processing PDF: {e}")
        print("\nMake sure you have installed poppler-utils with:")
        print("sudo apt-get update && sudo apt-get install -y poppler-utils")

if __name__ == "__main__":
    main()
