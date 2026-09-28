
from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "interview_started" not in st.session_state:
    st.session_state.interview_started = False


# ---------------------------------------------------------
# PROCESS DOCUMENT
# ---------------------------------------------------------

def process_document(path):

    # Load PDF files
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(docs)

    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    # Create vector database
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    st.session_state.vector_store = vector_db

    # -----------------------------------------------------
    # LLM
    # -----------------------------------------------------

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )

    # -----------------------------------------------------
    # RETRIEVAL TOOL
    # -----------------------------------------------------

    @tool
    def retrieve_context(query: str):
        """
        Search the uploaded resume and return relevant
        information from it.
        """

        docs = vector_db.similarity_search(
            query=query,
            k=4
        )

        context = ""

        for doc in docs:
            context += doc.page_content + "\n\n"

        return context

    # -----------------------------------------------------
    # SYSTEM PROMPT
    # -----------------------------------------------------

    system_prompt = """
You are an AI Interview Coach designed to help candidates
prepare for technical interviews and identify the skills
they need to improve for their career goals.

The candidate will upload their resume.

Carefully analyze the resume and understand:

- Education
- Technical skills
- Programming languages
- Projects
- Tools
- Frameworks
- Certifications
- Experience

Do not invent any skills, experience, projects,
certifications, or achievements that are not present
in the resume.

Clearly distinguish between:

1. Skills explicitly mentioned in the resume
2. Skills that appear weak or insufficiently demonstrated
3. Skills that are completely absent from the resume

Use the candidate's resume to create a personalized
technical interview.

Ask only ONE interview question at a time.

Start with questions appropriate to the candidate's
current level.

Gradually increase difficulty when the candidate
demonstrates strong understanding.

Questions may cover:

- Programming
- Computer Science fundamentals
- Projects
- APIs
- Databases
- AI/ML
- Generative AI
- LLMs
- RAG
- Embeddings
- Vector databases
- LangChain
- FastAPI
- Git
- Deployment

Only ask about technologies that are relevant to the
candidate's resume or career preparation.

After every answer:

1. Evaluate the answer objectively.
2. Explain what was correct.
3. Identify mistakes.
4. Identify missing concepts.
5. Give a concise assessment.
6. Suggest how the candidate can improve.
7. Ask the next interview question.

If the candidate repeatedly struggles with a concept,
ask another question about that concept from a different
angle.

If the candidate demonstrates strong understanding,
increase the difficulty.

At the beginning or end of the interview, provide a
personalized skill-gap analysis containing:

- Skills demonstrated in the resume
- Strong skills
- Weak skills
- Missing important skills
- Recommended topics
- Priority level
- Suggested projects or exercises

At the end of the interview, provide a performance
summary covering:

- Technical accuracy
- Clarity
- Depth of knowledge
- Problem-solving ability
- Overall interview readiness

Finally, provide a prioritized preparation roadmap.

Your primary goal is to simulate a realistic technical
interview while helping the candidate understand:

- What they know
- What they are missing
- Why those gaps matter
- What they should learn next
"""


    # -----------------------------------------------------
    # MEMORY
    # -----------------------------------------------------

    memory = InMemorySaver()

    # -----------------------------------------------------
    # CREATE AGENT
    # -----------------------------------------------------

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    # Save agent
    st.session_state.agent = agent

    # Mark document as uploaded
    st.session_state.document_uploaded = True


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------

st.title("NexPrep - Next Step in Your Career🎓")


# ---------------------------------------------------------
# PDF UPLOAD
# ---------------------------------------------------------

if not st.session_state.document_uploaded:

    uploaded = st.file_uploader(
        label="Select PDF File",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:

        with st.spinner("Processing documents..."):

            path = "./doc_files"

            os.makedirs(
                path,
                exist_ok=True
            )

            # Save uploaded PDFs
            for file in uploaded:

                file_path = os.path.join(
                    path,
                    file.name
                )

                with open(file_path, "wb") as f:
                    f.write(file.getvalue())

            # Process documents
            process_document(path)

        st.success("Documents processed successfully!")

        st.rerun()


# ---------------------------------------------------------
# DISPLAY PREVIOUS MESSAGES
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# START INTERVIEW
# ---------------------------------------------------------

if (
    st.session_state.document_uploaded
    and st.session_state.agent
    and not st.session_state.interview_started
):

    res = st.session_state.agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Read my resume carefully and ask me "
                        "the first interview question based "
                        "on it. Ask only one question at a time."
                    )
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": "interview_session_1"
            }
        }
    )

    # Get AI response
    ai_response = res["messages"][-1].content

    # Handle Gemini response format
    if isinstance(ai_response, list):

        text_parts = []

        for item in ai_response:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    text_parts.append(
                        item.get("text", "")
                    )

            elif isinstance(item, str):

                text_parts.append(item)

        ai_response = "".join(text_parts)

    # Save AI message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": ai_response
        }
    )

    # Mark interview as started
    st.session_state.interview_started = True

    # Display AI question
    with st.chat_message("assistant"):
        st.markdown(ai_response)


# ---------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------

user_input = st.chat_input("Your Answer...")


# ---------------------------------------------------------
# PROCESS USER ANSWER
# ---------------------------------------------------------

if user_input:

    # Save user answer
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Display user answer
    with st.chat_message("user"):
        st.markdown(user_input)

    # Send answer to agent
    res = st.session_state.agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": "interview_session_1"
            }
        }
    )

    # Get AI response
    ai_response = res["messages"][-1].content

    # Handle Gemini response format
    if isinstance(ai_response, list):

        text_parts = []

        for item in ai_response:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    text_parts.append(
                        item.get("text", "")
                    )

            elif isinstance(item, str):

                text_parts.append(item)

        ai_response = "".join(text_parts)

    # Save AI response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": ai_response
        }
    )

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(ai_response)


