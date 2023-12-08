# aisandpit

Setup and exmaples for running a local LLM to avoid expensive OpenAI API costs.

### install ollama and lamgchain

Ollama: https://ollama.ai/

Langchain: https://python.langchain.com/docs/integrations/llms/ollama


### Project Layout

Each folder should be a helpful example or key business problem to solve:


`/basic-exmaples`  simple LLM setup exmaples, start here to make sure your setup is working

`orchestra-scraping` experiemnt with extracting performance date/time/location/price from orchestra web pages ideally in json format

`clinc-scraping` experiment with extracting aesthetics treatments or doctor names from clinc web pages

`cantus-id` experiments with fetching cantus chants from fragments of annotated gregorian chant manuscripts




### Next steps and ideas to explore


##### LLM + RAG 
Curently solving the problems using LLM + RAG
@see Article: https://research.ibm.com/blog/retrieval-augmented-generation-RAG
@see RAG Apps: https://www.youtube.com/watch?v=TRjq7t2Ms5I
@see Vectoring words: https://www.youtube.com/watch?v=gQddtTdmG_8

The initial web scrape exmaples load a web page, split it up in to 1000(ish) token chucks, loads in to a vectorbase (embeddings), and then 
puts the vectodb docs in the prompt template, which in not ideal and has token length limitations. It works ok for the current problems
but we may need to better chunk up the data and filter what goes in, there's a lot of noise.

### LLM Storage

We'd like to find a solution for better large document storage that does not inject them in to the prompt. 


### Langchain

Would be good to see more use of lang chain tools/modules eg: web/database/document loaders etc...

ideas: 
1. use langchain to answwer a natural language query via a sql database (have had this working) 
2. use langchain to answwer a natural language query via multiple databases 
3. use langchain to answer questions about data in a csv, eg: statisical questions
4. use langchain to render graphs from data in csv file

### Agents/AGI

Langchain agents and Babgy AGI look really intersting, and would be good to get running with a local LLM to see how it could solve some of the problems we throw at it.

@see `./baby-agi`


### Function calling 

This is how the more interesting AI apps are created.

* Create chatbots that answer questions by calling external tools (e.g., like ChatGPT Plugins)
* Convert natural language into API calls or database queries
* Extract structured data from text

@see https://github.com/rizerphe/local-llm-function-calling
@see https://platform.openai.com/docs/guides/function-calling
