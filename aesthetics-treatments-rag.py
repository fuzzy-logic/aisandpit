from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# @see https://python.langchain.com/docs/integrations/llms/ollama
# setup:
# ./ollama serve
# ./ollama run llama2
# run: python aesthetics-treatments-rag.py

# SETUP LLM:
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     model_path="/Users/rlm/Desktop/Code/llama.cpp/models/llama-2-13b-chat.ggufv3.q4_0.bin",
#     n_gpu_layers=n_gpu_layers,
#     n_batch=n_batch,
#     n_ctx=2048,
#     f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
#     callback_manager=callback_manager,
#     verbose=True,
# )

# this uses the local llm web server apis once you have it running via ollma: https://ollama.ai/
llm = Ollama(
   model="llama2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)




# VECTORDB-IZE WEB DATA
pages = ["https://www.epsomskinclinics.com/"] # epsom skin clinic
# pages = ["https://www.altondental.co.uk/"]



for page in pages: 
    loader = WebBaseLoader(page)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())


print("data sourced from following web pages: ", pages)

CUSTOM_PROMPT = PromptTemplate.from_template(
     """You are looking aesthetics treatments offered. 
     Do not list any aesthetics treatments that are not in the document, if you can't find any treatments just say 'I could not find any treatments'. 
     Only use the name of the treatment, do not describe details of the treatment. 
     Use following documents: {docs}"""
)

# Chain
llm_chain = LLMChain(llm=llm, prompt=CUSTOM_PROMPT)

# Run
question = "list the names of aesthetics treatments found as a bullet point list. Do not include any other text"
docs = vectorstore.similarity_search(question)
result = llm_chain(docs)

# Output
print("QUESTION:\n", question)
print("RESPONSE:\n ", result["text"])




