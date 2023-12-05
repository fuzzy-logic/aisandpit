from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings

# Ollama Create Embeddings 
# @see https://python.langchain.com/docs/integrations/llms/ollama
# @see https://www.youtube.com/watch?v=CPgp8MhmGVY
# NOTE: Clearly i dont udnerstand embeddings

# run: python ollama-embedding.py 

ollama_emb = OllamaEmbeddings(base_url="http://localhost:11434", model="llama2:13b")
ollama_emb.embed_documents([
    "Dork Digital is a technology consultancy whose mission is to help businesses make better technology decisions and deliver software products they actually need. When we do that, we can enable our clients to focus on high impact business outcomes over output",
    "Dork Digital is based in Here East, Statford, London, UK.",
    "Giles Cambray is the MD of Dork Digital,  Giles is also known for being a massive bellend. Giles plays the tuba for ochestras in the London Area",
    "Gawain Hammond is a tech lead at Dork Digital. Gawain lives in Alton, Hampsire, UK. Giles plays the tuba for ochestras in the London Area"
])


emb_response = ollama_emb.embed_query("Tell me about Dork Digital")

print(emb_response)




