import json
import os
import uuid

import chainlit as cl
import cohere
import yaml
from chainlit import on_chat_start
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from openai import AsyncOpenAI
from pinecone import Pinecone

#Set Up client and environment
client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
co = cohere.ClientV2(os.environ['COHERE_API_KEY'])

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
    msg = cl.Message(content="")
    await msg.send()
    await cl.sleep(00000000000.1)
    # Call Cohere chat query gen mode
    try:
        instructions = (
            "Context: You are part of a Retrieval Augmented Generation (RAG) Conversational QA system. You are the search query generator. Generate a single search query that accurately reflects the user's intent. "
            "The output should simply be a search query, without any additional information or lists."
        )

        # Generate search queries 
        search_queries = []


        # Define the query generation tool
        # query_gen_tool = [
        #     {
        #         "type": "function",
        #         "function": {
        #             "name": "internet_search",
        #             "description": "Returns a list of relevant document snippets for a textual query retrieved from the internet",
        #             "parameters": {
        #                 "type": "object",
        #                 "properties": {
        #                     "queries": {
        #                         "type": "array",
        #                         "items": {"type": "string"},
        #                         "description": "a list of queries to search the internet with.",
        #                     }
        #                 },
        #                 "required": ["queries"],
        #             },
        #         },
        #     }
        # ]

        res = co.chat(
            model="command-a-03-2025",
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": message.content},  # Use message.content instead of message
            ],
            # tools=query_gen_tool
        )
        print("search query", res)

        search_query = res.message.content[0].text if res.message.content else message.content

        id = await msg.send()
        await task_list.add_task(cl.Task(title=f"Generated Search Query: {search_query}", status=cl.TaskStatus.DONE))

        if res.message.tool_calls:
            for tc in res.message.tool_calls:
                queries = json.loads(tc.function.arguments)["queries"]
                search_queries.extend(queries)
        print(search_queries)
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
    task2.status = cl.TaskStatus.DONE
    await task_list.send()

    # print('retrieved', retrieved)

    
    urls = list(set([d.metadata['source'] for d in retrieved]))
    if mode == "Learn Mode":
        docs = [{"Title": d.metadata['title'], "Content": d.page_content} for i, d in enumerate(retrieved)]
    else:
        docs = [{"Content": d.page_content} for i, d in enumerate(retrieved)]


    yaml_docs = [yaml.dump(doc, sort_keys=False) for doc in docs]

    
    task3 = cl.Task(title="Re-Ranking Results", status=cl.TaskStatus.RUNNING)
    await task_list.add_task(task3)
    await task_list.send()

    # Rerank the top results
    reranked = co.rerank(model="rerank-v3.5", query=search_query, documents=yaml_docs, top_n=5)

    reranked_docs = [
        {
            "data": {
                "title": docs[result.index]["Title"] if mode == "Learn Mode" else None,
                "snippet": docs[result.index]["Content"],
            }
        }
        for result in reranked.results
    ]
    # print("Rereanked", reranked_docs)
    task3.status = cl.TaskStatus.DONE
    await task_list.send()
    
    # Generate final response stream with cohere chat
    task4 = cl.Task(title="Generating Response", status=cl.TaskStatus.RUNNING)
    await task_list.add_task(task4)
    await task_list.send()
    try:
        # Define the messages list with the preamble and user message
        messages = [
        {"role": "system", "content": (
            "You are SPARK, a Prompt Assistant created by Conversational AI Expert - Amogh Agastya (https://amagastya.com)."
            "SPARK stands for Smart Prompt Assistant and Resource Knowledgebase. SPARK exudes a friendly and knowledgeable persona,"
            "designed to be a reliable and trustworthy guide in the world of prompt engineering."
            "There are two modes: 'Learn Mode' for generating informative responses and 'Prompt Mode' for crafting prompts."
            "In 'Prompt Mode', SPARK helps generate prompts for users based on their queries. It provides relevant information and resources to assist them in crafting effective prompts."
            "Additionally, SPARK in prompt mode can chat with the user to clarify and craft the best prompt for their objectiive. You can also provide reasoning behind the crafted prompt."
            f"The user is currently on mode {mode}"
        )},
            {"role": "user", "content": message.content}  
        ]

        stream = co.chat_stream(
            model="command-a-03-2025",
            messages=messages,
            documents=reranked_docs,
        )
        
        response_text = ""
        citations = []
        for chunk in stream:
            if chunk:
                if chunk.type == "content-delta":
                    response_text += chunk.delta.message.content.text
                    # print(chunk.delta.message.content.text, end="")
                    await msg.stream_token(chunk.delta.message.content.text)
                elif chunk.type == "citation-start":
                    citations.append(chunk.delta.message.citations)

        task4.status = cl.TaskStatus.DONE
        await task_list.send()

                
    except Exception as e:
        print(f"Error generating response: {e}")
                
    if mode != "Prompt Mode":  # Only display sources if not in prompt mode
        if mode == "Learn Mode":
            sources = "\n".join([f"- {url}" for url in urls])
        else:
            sources = "\n\n".join([doc['data']['snippet'] for doc in reranked_docs])  # Adjusted to match new structure

        await cl.Message(content=f"*Sources*:\n\n{sources}", parent_id=id).send()

    task4.status = cl.TaskStatus.DONE
    await task_list.send()
    
    task_list.status = "Completed Successfully"
    await task_list.send()