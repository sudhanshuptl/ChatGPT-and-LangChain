from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory, ConversationSummaryMemory
from dotenv import load_dotenv

load_dotenv("../.env")

chat = ChatOpenAI()

# this will keep track of all the messages but does not work well if size of chat is too big
# so in that case you need to move to summery the previous chat
# memory = ConversationBufferMemory(
#     # keep track of chat history in a json file
#     chat_memory=FileChatMessageHistory("messages.json"),
#     memory_key="messages",
#     return_messages=True
#     )


# It has internal chain to send request to llm to sumerise the previous chat
memory = ConversationSummaryMemory(
    memory_key="messages",
    return_messages=True,
    llm=chat,
    verbos=True # to visualise this internal call to summerise previous conversation
    )

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory
)

while True:
    content = input(">> ")

    result = chain({"content": content})

    print(result["text"])
