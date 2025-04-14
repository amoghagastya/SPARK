import os
import uuid

import chainlit as cl
import cohere
from chainlit import on_chat_start
from dotenv import load_dotenv
# from langsmith.run_helpers import traceable
from langchain_openai import OpenAIEmbeddings
# from langchain.embeddings import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from openai import AsyncOpenAI

load_dotenv()
#Set Up client and environment
# client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
co = cohere.ClientV2(os.environ['COHERE_API_KEY'])
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "SPARK"  # Optional: "default" is used if not set
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

#Initialize embeddings & vectorstore
# embeddings = CohereEmbeddings(cohere_api_key=os.environ['COHERE_API_KEY'], model="embed-english-light-v3.0")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

pc = Pinecone(
        api_key=os.environ['PINECONE_API_KEY']
    )

learn_index = pc.Index('sparklearn')
prompt_index = pc.Index('spark-prompts')

learnsearch = PineconeVectorStore(index=learn_index, embedding=embeddings)
promptsearch = PineconeVectorStore(index=prompt_index, embedding=embeddings)

learn_retriever = learnsearch.as_retriever(search_kwargs={"k": 8})
prompt_retriever = promptsearch.as_retriever(search_kwargs={"k": 8})

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Learn Mode",
            markdown_description="Use this mode to learn about prompt engineering.",
            icon="https://www.shutterstock.com/image-vector/brain-emoji-vector-isolated-faces-600nw-2344535053.jpg",
        ),
        cl.ChatProfile(
            name="Prompt Mode",
            markdown_description="Use this mode to query the prompt database.",
            icon="https://e7.pngegg.com/pngimages/296/768/png-clipart-emoji-memorandum-computer-icons-text-messaging-writing-writing-pencil-emoticon.png",
        ),
    ]

@on_chat_start
async def init():
    conversation_id = str(uuid.uuid4())
    cl.user_session.set("id", conversation_id)
    # pc = Pinecone(
    #     api_key=os.environ.get("PINECONE_API_KEY")
    # )

# @traceable(run_type="chain")       
@cl.on_message
async def main(message: cl.Message):    
    task_list = cl.TaskList()
    task_list.status = "Running..."
    
    mode = cl.user_session.get("chat_profile")

    # Create a task and put it in the running state
    task1 = cl.Task(title="Generating Search Query", status=cl.TaskStatus.RUNNING)
    await task_list.add_task(task1)
    await task_list.send()
    
    # Add 'running' loader in UI
    # msg = cl.Message(content="")
    # await msg.send()
    # await cl.sleep(00000000000.1)
    # Call Cohere chat query gen mode
    # Define the query generation tool
    query_gen_tool = [
        {
            "type": "function",
            "function": {
                "name": "internet_search",
                "description": "Returns a list of relevant document snippets for a textual query retrieved from the internet",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "a list of queries to search the internet with.",
                        }
                    },
                    "required": ["queries"],
                },
            },
        }
    ]

    # Define a system message to optimize search query generation
    instructions = "Write a search query that will find helpful information for answering the user's question accurately. If you need more than one search query, write a list of search queries. If you decide that a search is very unlikely to find information that would be useful in constructing a response to the user, you should instead directly answer."
    # Generate search queries (if any)
    search_queries = []





    try:
        # query = co.chat(
        #     message=message.content,
        #     conversation_id=cl.user_session.get("id"),
        #     search_queries_only=True
        #     )

        res = co.chat(
            model="command-a-03-2025",
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": message.content},
            ],
            tools=query_gen_tool,
        )
        # print("DEBUG", res)
        if res.message.tool_calls:
            for tc in res.message.tool_calls:
                queries = json.loads(tc.function.arguments)["queries"]
                search_queries.extend(queries)

                
        print('search queries',search_queries)
        search_query = search_queries[0] if search_queries else message.content
    except Exception as e:
        print(f"Error generating search query: {e}")
        search_query = message.content
    task1.status = cl.TaskStatus.DONE
    await task_list.send()

    task2 = cl.Task(title="Retrieving Contexts", status=cl.TaskStatus.RUNNING)
    await task_list.add_task(task2)
    await task_list.send()
    
    # Set retriever based on mode
    if mode == "Learn Mode":
        retriever = learn_retriever 
    elif mode == "Prompt Mode":
        retriever = prompt_retriever
              
    retrieved = retriever.invoke(search_query)
    print("rretrieved ", retrieved)
    task2.status = cl.TaskStatus.DONE
    await task_list.send()
    
    urls = list(set([d.metadata['source'] for d in retrieved]))
    docs = [{"text": d.page_content} for i, d in enumerate(retrieved)]
    
    task3 = cl.Task(title="Re-Ranking Results", status=cl.TaskStatus.RUNNING)
    await task_list.add_task(task3)
    await task_list.send()

    # Rerank the top results
    reranked = co.rerank(model="rerank-v3.5", query=search_query, documents=docs, top_n=4)
    print("reranked ", reranked)
    reranked_docs = []
    for doc in reranked.results:
        reranked_doc = {
            "title": "doc_" + str(doc.index),
            # "snippet": doc["text"], 
            "snippet": doc.document       
            }
        reranked_docs.append(reranked_doc)
            
    task3.status = cl.TaskStatus.DONE
    await task_list.send()
    
    # Generate final response stream with cohere chat
    task4 = cl.Task(title="Generating Response", status=cl.TaskStatus.RUNNING)
    await task_list.add_task(task4)
    await task_list.send()
    try:
        messages = [
            {"role": "system", "content": "You are SPARK, a Prompt Assistant created by Conversational AI Developer - Amogh Agastya (https://amagastya.com). SPARK stands for Smart Prompt Assistant and Resource Knowledgebase. SPARK exudes a friendly and knowledgeable persona, designed to be a reliable and trustworthy guide in the world of prompt engineering. If user requests for a prompt, make sure to ALWAYS enclose the prompt with triple backticks ```. It is very important that you format the prompts in ```, so make sure to adhere to that at all costs."},
            {"role": "user", "content": message.content}
        ]
        stream = co.chat_stream(
                # conversation_id=cl.user_session.get("id"),
                messages=messages,
                documents=reranked_docs,
                model='command-a-03-2025',
                # stream=True,
                # prompt_truncation='AUTO',
                # temperature=0.5
        )
        
        msg = cl.Message(content="")
        await msg.send()       
        
        for chunk in stream:
            # if event.event_type == "text-generation":   
            #     await msg.stream_token(event.text)
            # if event.event_type == "stream-end":
            #     break

            if chunk:
                if chunk.type == "content-delta":
                    response_text += chunk.delta.message.content.text
                    print(chunk.delta.message.content.text, end="")
                    await msg.stream_token(chunk.delta.message.content.text)

                
    except Exception as e:
        print(f"Error generating response: {e}")
                
    # Send and close the message stream
    id = await msg.send()
    await cl.Message(content=f"Generated Search Query: {search_query}", parent_id=id).send()
    if mode == "Learn Mode":
        sources = "\n".join([f"- {url}" for url in urls])
    else:
        sources = "\n\n".join([doc['snippet'] for doc in reranked_docs])
    await cl.Message(content=f"*Sources*:\n\n{sources}", parent_id=id).send()
    
    task4.status = cl.TaskStatus.DONE
    await task_list.send()
    
    task_list.status = "Completed Successfully"
    await task_list.send()
