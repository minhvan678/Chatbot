import streamlit as st
from langchain import hub
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, AIMessageChunk
from langchain_core.documents import Document
from langchain_core.runnables import RunnableBranch
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from typing import List, Sequence
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
web_search_tool = TavilySearchResults()

web_search_decision_prompt = hub.pull("john-chatly/web_search_necessity_classifier")

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

### Answer question ###
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context and chat history to answer the question. If you don't know the answer, say that you don't know."
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)


### Statefully manage chat history ###
class GraphState(TypedDict):
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    answer: str
    documents: List[str]

class Chatbot:
    def __init__(self, db, llm):
        self.db = db
        self.llm = llm
        self.chain = qa_prompt | llm.with_config(callbacks=[StreamingStdOutCallbackHandler]) | StrOutputParser()
        self.retriever = db.as_retriever(search_type="similarity_score_threshold",search_kwargs={'score_threshold': 0.5, 'k': 2})
        self.web_search_decision = web_search_decision_prompt | llm
    ### Nodes
    def retrieve(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """

        question = state["question"]
        chat_history = state["chat_history"]
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        # Retrieval
        documents = history_aware_retriever.invoke({"input": question, "chat_history": chat_history})
        return {"documents": documents, "question": question}

        
    def web_search(self, state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        question = state["question"]

        # Web search
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)

        return {"documents": web_results, "question": question}

    def generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, answer, that contains LLM generation
        """

        question = state["question"]
        documents = state["documents"]
        chat_history = state["chat_history"]
        # RAG generation
        generation = self.chain.invoke({"context": documents, "question": question, "chat_history": chat_history})
        return {"chat_history": [HumanMessage(question), AIMessage(generation)], "answer": generation}


    ### Edges

    def route_question(self, state):
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        documents = state["documents"]
        question = state["question"]
        if not documents:
            decision = self.web_search_decision.invoke({"input": question}).content

            if decision=='N':
                ("---ROUTE QUESTION TO GENERATE---")
                return "generate"
            else:

                return "web_search"
        else:

            return "vectorstore"

    def create_app(self):

        workflow = StateGraph(GraphState)
        # Define the nodes
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("web_search", self.web_search)  # web search
        workflow.add_node("generate", self.generate)  # generatae


        # Build graph
        workflow.add_edge(START, "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self.route_question,
            {
                "web_search": "web_search",
                "vectorstore": "generate",
                "generate": "generate"
            },
        )

        workflow.add_edge("web_search", "generate")
        
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        return app


def stream_output_to_streamlit(question: str, app, config):
    # Create a placeholder for the output
    output_placeholder = st.empty()
    response = ""  # Initialize an empty response to store the streamed chunks
    inputs = {"question": question}

    # Stream the output chunks to the Streamlit UI
    for chunk, metadata in app.stream(inputs, config=config, stream_mode="messages"):
        if isinstance(chunk, AIMessageChunk) and metadata["langgraph_node"] == "generate" :  # Check if the chunk is an AI message
            # Append the content to the response
            response += chunk.content
            # Update the Streamlit placeholder with the latest output
            output_placeholder.markdown(response)

    return response