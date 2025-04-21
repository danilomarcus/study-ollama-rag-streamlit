import os
from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

pdf_path = '/home/danilo/Develops/ai/dados.pdf'

# Configurar modelos - otimizados para desenvolvimento de software
embedding_model = "mxbai-embed-large"  # Melhor modelo para embeddings técnicos
rag_model = "llama3.2"  # Ou "codellama:instruct" se disponível

# Verifica se o arquivo existe
if not os.path.exists(pdf_path):
    print(f'PDF file not found: {pdf_path}')
    exit()

try:
    # Carrega o documento PDF
    loader = UnstructuredPDFLoader(file_path=pdf_path)
    print("Loading PDF...")
    data = loader.load()
    print("Done loading!")
    print(f"Type of data: {type(data)}")

    # Processa o conteúdo
    content = data[0].page_content
    
    # Divisão em chunks otimizada para documentação técnica
    # Chunks maiores para preservar contexto de requisitos e código
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,      # Chunks maiores para capturar contexto técnico
        chunk_overlap=300,    # Overlap para manter referências cruzadas
        separators=["\n\n", "\n", ".", " ", ""],  # Priorizar quebra em parágrafos
    )
    chunks = text_splitter.split_documents(data)
    print(f"Split into {len(chunks)} chunks")
    # print(f"First chunk: {chunks[0]}")

    # Inicializa o modelo de embedding
    import ollama
    print(f"Pulling embedding model: {embedding_model}")
    ollama.pull(embedding_model)
    print("Done pulling embedding model!")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=embedding_model),
        collection_name="software-dev-rag",
    )
    print("Done creating vector database!")

    # ==== Retrieval ====
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_ollama import ChatOllama
    from langchain_core.runnables import RunnablePassthrough
    from langchain.retrievers.multi_query import MultiQueryRetriever

    # Configurando o modelo de linguagem
    print(f"Configurando modelo RAG: {rag_model}")
    llm = ChatOllama(model=rag_model)

    # Prompt para gerar repostas coerentes - otimizado para terminologia técnica
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
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

    # Consultando o RAG
    # res = chain.invoke(input="Quais são os stakeholders citados no documento?")
    # res = chain.invoke(input="Quem é o responsável pela criação do documento, existe um controle de versão?")
    # res = chain.invoke(input="Do que se trata o documento?")
    res = chain.invoke(input="Do que se trata o documento? Identifique requisitos principais e possíveis desafios técnicos.")
    print("\n\nResposta do RAG:")
    print("-" * 50)
    print(res)
    print("-" * 50)

except Exception as e:
    print(f"Error found: {e}")

