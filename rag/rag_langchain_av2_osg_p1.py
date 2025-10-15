import os
import time
import bs4
import argparse
import json

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_langchain_helper import *

parser = argparse.ArgumentParser()
parser.add_argument('--llm_type', type=str, default='openai')
parser.add_argument('--llm_model', type=str, default='gpt-4o')
parser.add_argument('--map_name', type=str, default='map_nissan_small')
parser.add_argument('--api_path', type=str, default='../openai_api.txt')
args = parser.parse_args()

if args.llm_type == 'openai':
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
    embedding = OpenAIEmbeddings()
    with open(args.api_path, 'r') as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()

elif args.llm_type == 'llama':
    from langchain_ollama import OllamaLLM
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding = HuggingFaceEmbeddings()

else:
    print('Model not found')
    exit()


print('map_name', args.map_name)
print('llm_model', args.llm_model)

persist_directory = f"./chroma_db_{args.llm_type}_{args.llm_model}"


# Check if the vectorstore already exists
if not os.path.exists(persist_directory):
    # Load, chunk and index the contents of the PDF.
    start = time.time()
    loader = PyPDFLoader(
        file_path="../hdm-chap-300.pdf",
        # extract_images = True,
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create and persist the vectorstore
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embedding,
        persist_directory=persist_directory
    )
    print("Vectorstore created and saved.")
    print("Time taken for creating the vectorstore: {:.2f} seconds".format(time.time() - start))

else:
    # Load the existing vectorstore
    start = time.time()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    print(f"Existing vectorstore loaded. {persist_directory}")
    print("Time taken for loading the vectorstore: {:.2f} seconds".format(time.time() - start))


# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(search_type="mmr")
prompt = hub.pull("rlm/rag-prompt")

llm_model = args.llm_model

if args.llm_type == 'openai':
    llm = ChatOpenAI(model=llm_model)
elif args.llm_type == 'llama':
    llm = OllamaLLM(model=llm_model)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

map_name = args.map_name
with open('../outputs/av2/{}_output.json'.format(map_name), 'r') as f:
    json_data = json.load(f)

road_list = [road[1] for road in json_data["streets"]["roads"]]

basic_road_info = get_basic_road_info(road_list)
with open('../outputs/av2/{}_basic_road_info.json'.format(map_name), 'w') as f:
    json.dump(basic_road_info, f, indent=2)

basic_road_info_str = json.dumps(basic_road_info)

query = basic_road_info_str + "\n Given the above basic road information in JSON format, return the detailed road information in the following JSON output format: \n\
{\n\
    id: {\n\
        'name': ,\n\
        'lane_width': type=int; lane width in feet for the given road type,\n\
        'bike_lane_width': type=int; bike lane width in feet for the given road type,\n\
    }\n\
}\n\
Please provide the output strictly in JSON format with no comments and no explanations."

start = time.time()
answer = rag_chain.invoke(query)
print("Time taken for generating the answer: {:.2f} seconds".format(time.time() - start))

retrieved_docs = retriever.invoke(query)
print_retrieved_docs(retrieved_docs, output_file='../outputs/av2/{}_{}_{}_retrieved_docs_osg_p1.txt'.format(map_name, args.llm_type, llm_model))

formatted_answer = answer.replace("'", '"').replace('```', '').replace('\n', '').replace('json', '')
json_output = json.loads(formatted_answer)

for k, v in json_output.items():
    if "name" not in v:
        v["name"] = None
if "bike_lane_width" not in v or v["bike_lane_width"] is None:
    v["bike_lane_width"] = 0
if "lane_width" not in v or v["lane_width"] is None:
    v["lane_width"] = 0


with open('../outputs/av2/{}_road_info_{}_{}.json'.format(map_name, args.llm_type, llm_model), 'w') as f:
    json.dump(json_output, f, indent=2)