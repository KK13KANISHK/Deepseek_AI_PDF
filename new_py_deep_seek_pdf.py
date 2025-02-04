try:
    import gradio as gr
    import re
    import tempfile
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma
    from langchain_community.embeddings import OllamaEmbeddings
    import ollama
except ImportError as e:
    print(f"Missing required package: {str(e)}")
    exit(1)

import logging
import atexit

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_interpreter.log'),
        logging.StreamHandler()
    ]
)

MODEL_NAME = "deepseek-r1"

def check_system_health():
    try:
        # Check if Ollama is running
        ollama.list()
        
        # Check if model is available
        models = ollama.list()
        if MODEL_NAME not in str(models):
            print(f"Warning: {MODEL_NAME} model not found")
            return False
        
        return True
    except Exception as e:
        print(f"System health check failed: {str(e)}")
        return False

if not check_system_health():
    print("System health check failed. Please ensure all components are running.")
    exit(1)

def process_pdf(pdf_file):
    logging.info(f"Processing new PDF file")
    if pdf_file is None:
        return None, None, None
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.read())
        temp_pdf_path = temp_pdf.name
    
    loader = PyMuPDFLoader(temp_pdf_path)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    return text_splitter, vectorstore, retriever

def combine_docs(docs, max_length=4000):
    combined = "\n\n".join(doc.page_content for doc in docs)
    if len(combined) > max_length:
        return combined[:max_length] + "..."
    return combined

def ollama_llm(question, context):
    try:
        formatted_prompt = f"Question: {question}\n\nContext: {context}"
        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'user', 'content': formatted_prompt}
        ])
        response_content = response['message']['content']
        final_answer = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
        return final_answer
    except Exception as e:
        return f"Error generating response: {str(e)}"

def rag_chain(question, text_splitter, vectorstore, retriever):
    retrieved_docs = retriever.invoke(question)
    formatted_content = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_content)

def ask_question(pdf_file, question):
    text_splitter, vectorstore, retriever = process_pdf(pdf_file)
    if text_splitter is None:
        return "No PDF uploaded."
    result = rag_chain(question, text_splitter, vectorstore, retriever)
    return result

def cleanup_on_exit():
    logging.info("Cleaning up resources on exit")
    # Add any cleanup needed, e.g., closing database connections, deleting temp files

atexit.register(cleanup_on_exit)

interface = gr.Interface(
    fn=ask_question,
    inputs=[
        gr.File(
            label="Upload PDF (optional)",
            file_types=[".pdf"],
            file_count="single"
        ),
        gr.Textbox(
            label="Ask a question",
            placeholder="Enter your question about the PDF...",
            lines=2
        )
    ],
    outputs=gr.Textbox(
        label="Answer",
        lines=5
    ),
    title="PDF Question Answering System",
    description="""
    Upload a PDF document and ask questions about its content.
    The system will use DeepSeek-R1 to provide relevant answers.
    
    Limitations:
    - Maximum file size: 10MB
    - Supported format: PDF only
    """,
    allow_flagging="never",
    theme="default"
)

# Add error handling for the launch
try:
    interface.launch(share=False, debug=True)
except Exception as e:
    print(f"Failed to launch interface: {str(e)}")
