import streamlit as st

import os

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

# os.environ["OPENAI_API_KEY"]=st.secrets["OPENAI_API_KEY"]
# os.environ["PINECONE_API_KEY"]=st.secrets["PINECONE_API_KEY"]
# os.environ["PYTHONPATH"]=st.secrets["PYTHONPATH"]

from backend.core_with_memory import run_llm_with_memory

from typing import Set

from streamlit_chat import message


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""

    sources_list = list(source_urls)

    sources_list.sort()

    sources_string = "sources:\n\n"

    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}: [Source]{source}\n\n"

    return sources_string


st.header("LangChain Documentation Helper")

prompt = st.text_input("Prompt", placeholder="Enter your prompt about LangChain here ... ")    

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

if prompt:
    if prompt in st.session_state["user_prompt_history"]: 
        formatted_response=f'The prompt: "{prompt}" has already been entered this session.\nPlease enter a new prompt or restart the session (reload the page).'
        # st.session_state["chat_answers_history"].append(formatted_response)
        # st.session_state["user_prompt_history"].append(prompt)
        st.write(formatted_response)
    else:
        with st.spinner("Generating Response ... "):

            generated_response = run_llm_with_memory(
                query=prompt, chat_history=st.session_state["chat_history"]
            )

            sources = set(
                [doc.metadata["source"] for doc in generated_response["source_documents"]]
            )

            sources_string = create_sources_string(sources)

            formatted_response = f'{generated_response["result"]}\n\nSources:'
            for i, source in enumerate(sources):
                formatted_response += f"\n\n[Source {i+1}]({source})"
            # formatted_response += '\n'.join([f'[Source {i+1}]({source})' for i, source in enumerate(sources)])

            if prompt not in st.session_state["user_prompt_history"]:
                st.session_state["user_prompt_history"].append(prompt)

            st.session_state["chat_answers_history"].append(formatted_response)

            st.session_state["chat_history"].append(
                ("human", prompt)
            )  # LangChain likes/prefers tuples

            st.session_state["chat_history"].append(
                ("ai", generated_response["result"])
            )  # LangChain likes/prefers tuples

            # st.write(formatted_response)

if st.session_state["chat_answers_history"]:
    for user_query, generated_response in zip(
        st.session_state["user_prompt_history"],
        st.session_state["chat_answers_history"],
    ):
        message(user_query, is_user=True)
        message(generated_response, is_user=False)
