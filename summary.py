import os
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv() ## loading all the environment variable

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

def get_llm(use_llm='gpt'):
    if use_llm == "groq":
        llm=ChatGroq(model="qwen-2.5-32b")
    elif use_llm == "gpt":
        llm=ChatOpenAI(model="gpt-4o-mini")
    else:
        raise ValueError("Invalid LLM choice")
    return llm

def load_docs(pdf_url):
    def load_pdf(url):
        return PyPDFLoader(url).load()

    # Use number of CPU cores for optimal parallelization
    max_workers = min(5, len(pdf_url))
    print(f"Spawning {max_workers} threads")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        docs = list(executor.map(load_pdf, pdf_url))

    return docs

def create_retriever(docs):
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs_list)

    ## Add all these text to vectordb
    vectorstore=FAISS.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings()
    )
    retriever=vectorstore.as_retriever()
    return retriever

def create_tool(retriever):
    retriever_tool=create_retriever_tool(
        retriever,
        "retriever_vector_db_blog",
        "Search and retrieve information from the vector database"
    )
    tools=[retriever_tool]
    return tools

def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    llm=get_llm()
    llm = llm.bind_tools(tools)
    response = llm.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM with tool and validation
    llm=get_llm()
    llm_with_tool = llm.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    # print(messages)
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content
    # print("docs", docs)

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        # print(score)
        return "rewrite"
    
def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated message
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatGroq(model="qwen-2.5-32b")

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    llm=get_llm()
    response = llm.invoke(msg)
    return {"messages": [response]}

def create_graph(retriever_tool):
    workflow = StateGraph(AgentState)

    # Define the nodes we will cycle between
    workflow.add_node("agent", agent)  # agent
    retrieve = ToolNode(retriever_tool)
    workflow.add_node("retrieve", retrieve)  # retrieval
    workflow.add_node("rewrite", rewrite)  # Re-writing the question
    workflow.add_node(
        "generate", generate
    )  # Generating a response after we know the documents are relevant
    # Call agent node to decide to retrieve or not
    workflow.add_edge(START, "agent")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "agent",
        # Assess agent decision
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents,
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    # Compile
    graph = workflow.compile()
    return graph


if __name__ == "__main__":
    pdf_url=[
        'USFDA/AMTAGVI.pdf',
        'USFDA/BENOQUIN.pdf',
        'USFDA/BRAFTOVI.pdf',
        'USFDA/COTELLIC.pdf',
        'USFDA/HEPZATO.pdf',
        'USFDA/INTRON_A.pdf',
        'USFDA/MEKTOVI.pdf',
        'USFDA/OPDIVO.pdf',
        'USFDA/PROLEUKIN.pdf',
        'USFDA/TAFINLAR.pdf',
        'USFDA/TAFINLAR_2.pdf',
        'USFDA/TECENTRIQ.pdf',
        'USFDA/TECENTRIQ_HYBREZA.pdf',
        'USFDA/YERVOY.pdf',
    ]
    docs=load_docs(pdf_url)
    retriever=create_retriever(docs)
    tools=create_tool(retriever)
    graph=create_graph(tools)

    while True:
        question = input("Enter your question (or 'q' to exit): ")
        if question.lower() == 'q':
            break
            
        response = graph.invoke({"messages": question})
        messages = response.get("messages")

        for m in messages:
            print(m.pretty_print())