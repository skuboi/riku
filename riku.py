############################################################
############################################################
# RIKU BOT #
############################################################    
############################################################


############################################################
# PROJECT OVERVIEW #
############################################################    
    # 1. get the user query
    # 2. use the retriever to find the most similar document
    # 3. give the most similar document to the chat model
    # 4. use the chat model to generate a response
    # 5. return the response to the user
    # 6. repeat

############################################################
# IMPORTS #
############################################################
import streamlit as st
# from streamlit_chat import message
import langchain
from langchain.tools import Tool
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import CSVLoader, DirectoryLoader, TextLoader, PythonLoader, UnstructuredFileLoader
from langchain_community.document_loaders import UnstructuredAPIFileLoader
from langchain.agents import create_tool_calling_agent, create_openai_tools_agent, AgentExecutor
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
import pandas as pd
import os
from dotenv import load_dotenv
import openpyxl
from langchain_community.embeddings import OpenAIEmbeddings

############################################################
# KEYS #
############################################################
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

############################################################
# LOAD REFERENCE DATA #
############################################################
# Going to load the data with pre-chunked text files, all in the same directory
loader = DirectoryLoader('/Users/kuboi/riku/riku/chunks_1', glob="**/*.txt", loader_cls=UnstructuredFileLoader)
data = loader.load()


#The CSVLoader generates one document per row in th CSV, that document is generated using all columns of the corresponding row.
    # encoding: https://stackoverflow.com/questions/22216076/unicodedecodeerror-utf8-codec-cant-decode-byte-0xa5-in-position-0-invalid-s
    # source column: giving it "Title", which is the column name in the csv file (e.g. for classes data, it'll output "Flash Fencer", "Full Metal Jaguar", etc.)
    # consider adding this fucntionality later: https://github.com/langchain-ai/langchain/issues/6961

#loader = CSVLoader(file_path='/Users/kuboi/fandom-chatbot/ScrapeFandom-main/class_data_csv_0.csv', encoding = 'unicode_escape', source_column = "Title", csv_args={'delimiter': ','})

# print some examples
    #print("printing a new chunk")
    #print(data[0:4])


############################################################
# AI PROMPTS #
############################################################

# Define the system message template
system_template = """You are a helpful chatbot that answers a gamer's question about the video game Xenoblade Chronicles 3. 
If the user is asking about a in-game Class, be sure to reference the info provided by "CRITICAL_INFO" in the title.
You are given references and context, and you are expected to provide accurate and helpful answers.
If you don't know the answer, just say so.
Make sure any content you output is legible to a regular human.
Unless necessary, keep your answers short and to the point. Should be able to read the answer within 30 seconds for most answers.
Use the given references and context below the line to answer the users question. 
----------------
{context}"""

# Create the chat prompt templates
    # https://github.com/langchain-ai/langchain/issues/5462
prompt_bits = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
qa_prompt = ChatPromptTemplate.from_messages(prompt_bits)

############################################################
# EMBEDDINGS & VECTOR STORE #
############################################################
# OpenAI Embeddings
    # the data we imported gets split into vectors, which will index each row in our file
embeddings = OpenAIEmbeddings()


# Vector store
    # the vector store is the actual index of our data
    # it's the thing that we'll use to retrieve documents
    # when we get the USER query, we can use the vector store to find the most similar document
    # them we give the most similar docs to the LLM, which will use that to answer the question
    # for this we're using FAISS: 'library for efficient similarity search and clustering of dense vectors.'
        # https://github.com/facebookresearch/faiss
vectorstore = FAISS.from_documents(data, embeddings)

############################################################
# RETRIEVER #
############################################################
    # the retriever is the thing that we'll use to find the most similar document to the user's query
retriever = vectorstore.as_retriever(search_type='mmr')

retriever_tool = create_retriever_tool( # this will let us use the retriever in the chatbot as an agent tool
    retriever,
    "game_data_search",
    "Search for relevant Xenoblade Chronicles 3 game data. For any questions about Xenoblade Chronicles 3, use this tool!",
)

tools = [retriever_tool] # this just saves the tools we'll use downstream

# System
llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo-0125')
#llm var is used downstream by the ConversationalRetrievalChain, ConversationSummaryBufferMemory, and create_tool_calling_agent

############################################################
# STREAMLIT STATE & UI #
############################################################
st.set_page_config(
    page_title="RIKU: Real-time Interactive Knowledge Utility",
    page_icon=":rocket:",
    layout="wide",
)


st.title("RIKU BOT: Real-time Interactive Knowledge Utility")
st.write("Here to help you with Xenoblade Chronicles 3!")
st.divider()




# Establish statefulness for streamlit vars
    # messages will keep track of back-and-forth Q&As, to pass to GPT for context if there are follow-up questions
    # generated will keep track of the generated responses, to display to the user
    # past will keep track of the past responses, to display to the user
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'bot', 'content': "hello! you can ask me questions about classes, builds, skills, arts, and gems in xenoblade 3!", 'avatar': 'https://i.imgur.com/RWOuYxO.png'}]

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar = message["avatar"]):
        st.markdown(message["content"])

############################################################
# LANGCHAIN #
############################################################

# Establish langchain memory
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
@st.cache_resource
def init_memory():
    return ConversationSummaryBufferMemory(
        llm = llm,
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)
memory = init_memory()

# Conversational Retrieval Chain
    # The Conversational Retrieval Chain is a class that combines a retriever and a chat model into a single object.
    # It's the thing that we'll use to actually chat with the user.
    # allows the chatbot to have 'memory'
    # FIRST, combines the chat history (either explicitly passed in or retrieved from the provided memory) and the question into a standalone question
        # SECOND, looks up relevant documents from the retriever
        # THIRD, pass those documents and the question to a question answering chain
        # FOURTH, return a response
    # https://python.langchain.com/docs/use_cases/question_answering/how_to/chat_vector_db

chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever=retriever,
    return_source_documents=True,
    verbose=True,  
    get_chat_history=lambda h : h,
    memory = memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

# Store and share chat history
    # question will refer to what the user is CURRENTLY asking
    # we also provide the history, which gives context on the past parts of the conversation
    # we save this round of Q&A to the 'history' list

def conversational_chat(query):
    # Retrieve the most relevant documents using the encoded query
    result = chain({"question": query, 
    "chat_history": st.session_state['messages']
    })
    # st.session_state['messages'].append((query, result["answer"]))
        
    return result["answer"]


############################################################
# AGENT SET UP #
############################################################

# # Create agent
# prompt = hub.pull("hwchase17/openai-functions-agent")
# agent = create_tool_calling_agent(llm, tools,prompt)
# # we can initialize the agent with the LLM, the prompt, and the tools. The agent is responsible for taking in input and deciding what actions to take.

# # Create executor
# # combine the agent (the brains) with the tools inside the AgentExecutor (which will repeatedly call the agent and execute tools).
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


############################################################
# CHAT UI #
############################################################
    # this version will use the chat elements officially launched by streamlit team
        # https://docs.streamlit.io/library/api-reference/chat/st.chat_message
    # create a container that can house all the content with `chat_message`

    # display the history of messages when the app re-runs; stored in `messages`
#for msg in st.session_state['messages']:
    #with st.chat_message(msg['role'], avatar=msg['avatar']):
     #   st.markdown(msg['content']) 

    # create a text input field for the user to enter their question
if question := st.chat_input("ask your questions here"):
    with st.chat_message("user", avatar='https://i.imgur.com/xWRPp9m.png'):
        st.markdown(question)
        #agent_executor.invoke({"input": question})
    st.session_state.messages.append({'role': 'user', 'content': question, 'avatar': 'https://i.imgur.com/xWRPp9m.png'})
        # call OpenAI for a response from the chatbot
    with st.spinner("thinking..."):
        answer = conversational_chat(question)
        with st.chat_message("bot", avatar='https://i.imgur.com/RWOuYxO.png'):
            st.markdown(answer)
        st.session_state.messages.append({'role': 'bot', 'content': answer, 'avatar': 'https://i.imgur.com/RWOuYxO.png'})
    # print(st.session_state.messages)
        # reset state if needed, clear memory


if st.button('Switch to a new topic? This will clear your chat history.'):
    with st.spinner("ok, clearing context..."):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.session_state.clear()
        st.session_state['messages'] = []
        # st.session_state['messages'] = [{'role': 'bot', 'content': "ok, ready for your next topic question", 'avatar': '🤖'}]
        #st.session_state.messages.append({'role': 'bot', 'content': 'ok ready for next q', 'avatar': '🤖'})

    # for message in memory['chat_history']:
        #   del message
        # st.session_state['messages'] = []
        #memory.clear()
        #memory = init_memory()
        st.cache_resource.clear()
        # message = st.chat_message('bot', avatar='🤖')
        # message.markdown('ok! switched contexts, ask away!')
        if 'messages' not in st.session_state:
            st.session_state['messages'] = [{'role': 'bot', 'content': "hello! you can ask me questions about classes, builds, skills, arts, and gems in xenoblade 3!", 'avatar': 'https://i.imgur.com/RWOuYxO.png'}]

        with st.chat_message("bot", avatar='https://i.imgur.com/xWRPp9m.png'):
            st.markdown('context cleared - ready to move on to your next question!')

        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar = message["avatar"]):
                st.markdown(message["content"])
    

        # clear the langchain memory so we do not have chat history


