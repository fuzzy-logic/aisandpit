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
# run: python clinic-docs-rag.py 

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
pages = ["https://www.sknclinics.co.uk/about-skn/expert-medical-team", "https://www.sknclinics.co.uk/treatments-and-pricing"];


for page in pages: 
    loader = WebBaseLoader(page)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())


print("data sourced from following web pages: ", pages)






# Prompt
prompt = PromptTemplate.from_template(
    "Find people in the folllwing documents: {docs}"
)

# Chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Run
question = "Find names and job titles of doctors and nurses who work at the clinic"
docs = vectorstore.similarity_search(question)
result = llm_chain(docs)

# Output

result["text"]



