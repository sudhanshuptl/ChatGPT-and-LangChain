from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import  OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores.chroma import  Chroma

load_dotenv("../.env") 

# first look for 200 charactor then find the nearest seperator and break it
text_splitter = CharacterTextSplitter(
    separator="\n", #sperate the data on \n
    chunk_size=200, # approx size of single chunk
    chunk_overlap=100 # use last n charactor from previous seperator
)

# # generate embeddings via chatGPT embedding
# embeddings = OpenAIEmbeddings()
# emb = embeddings.embed_query("this is test message")
# print(emb)

transformer_embeddings = SentenceTransformerEmbeddings()
emb = transformer_embeddings.embed_query("HI there")
print(emb)

# load the text data and convert it in docs formate, you can also use pdfloader , etc as well
# loader = TextLoader("facts.txt")
# docs = loader.load_and_split(text_splitter=text_splitter)
#
# # for doc in docs:
# #     print(doc, end="\n\n")
#
# db = Chroma.from_documents(
#     docs,
#     embedding=embeddings,
#     persist_directory="emb"
# )