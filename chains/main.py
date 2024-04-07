from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain import LLMChain  
from langchain.chains import SequentialChain
from dotenv import load_dotenv
from pprint import pprint
import os

# Load .env file to env
load_dotenv("../.env")

llm = OpenAI()
# openai_api_key=os.environ.get("OPENAI_API_KEY") by default pic env set with OPENAI_API_KEY


code_prompt = PromptTemplate(
    template="write a very sort {langauge} function that will {task}",
    input_variables=["langauge", "task"]
)

generate_test_for_result = PromptTemplate(
    template="write a test for the following {langauge}  code:\n {code}",
    input_variables=["langauge", "code"]
)



code_chain = LLMChain(
    llm = llm,
    prompt = code_prompt,
    output_key = "code" # key to extract from the output
)

test_result_chain = LLMChain(
    llm = llm,
    prompt = generate_test_for_result,
    output_key = "test" # key to extract from the output
)


# result of code_chain is passed to test_result_chain
chain = SequentialChain(
    chains=[code_chain, test_result_chain],
    input_variables=["langauge", "task"],
    output_variables=["code", "test"]
    ) 

# result = code_chain({
#     "langauge": "python",
#     "task": "find the maximum number in a list"
# })

result = chain({
    "langauge": "python",
     "task": "find the maximum number in a list"

})
pprint(result)