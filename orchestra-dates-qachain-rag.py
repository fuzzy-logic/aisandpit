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
from langchain import hub
from langchain.chains import RetrievalQA


# @see https://python.langchain.com/docs/integrations/llms/ollama
# setup:
# ./ollama serve
# ./ollama run llama2
# run: python orchestra-dates-qachain-rag.py

# SETUP LLM:
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# this uses the local llm web server apis once you have it running via ollma: https://ollama.ai/
llm = Ollama(
    model="llama2",
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

# VECTORDB WEB DATA
pages = ["https://www.rpo.co.uk/whats-on/eventdetail/1982/82/john-rutters-christmas-celebration-matinee"];


for page in pages: 
    loader = WebBaseLoader(page)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
print("data sourced from following web pages: ", pages)

# This sets up a Question/Answer prompt in a chain that includes the prompt, embedding vector db, and the LLM:

# rag qa prompt info: https://smith.langchain.com/hub/rlm/rag-prompt-llama
# changing this prompt will radically change the behavior of the llm
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")


 
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)



# Run: this prompt is the instruction:
# multi event list Prompt: "List all performance events, include name, time, location, next performance date and any supplimental information that is provided"
# simple primary event prompt: "List the primaray performance event information. Include name, time, location, next performance date and any supplimental information that is provided"

question = "Output the primaray performance event name, date, time, location and supplimental information"
qa_chain({"query": question})





