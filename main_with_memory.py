OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]

from backend.core_with_memory import run_llm_with_memory

from typing import Set

import streamlit as st

from streamlit_chat import message


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""

    sources_list = list(source_urls)

    sources_list.sort()

    sources_string = "sources:\n\n"

    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}: {source}\n\n"

    return sources_string


st.header("LangChain Documentation Helper Application")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here ... ")

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

if prompt:
    with st.spinner("Generating Response ... "):

        generated_response = run_llm_with_memory(query=prompt, chat_history=st.session_state["chat_history"])

        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        formatted_response = (
            f'{generated_response["result"]}\n\n {create_sources_string(sources)}'
        )

        st.session_state["user_prompt_history"].append(prompt)

        st.session_state["chat_answers_history"].append(formatted_response)

        st.session_state["chat_history"].append(("human", prompt)) # LangChain likes/prefers tuples

        st.session_state["chat_history"].append(("ai", generated_response["result"])) # LangChain likes/prefers tuples

        # st.write(formatted_response)

if st.session_state["chat_answers_history"]:
    for user_query, generated_response in zip(
        st.session_state["user_prompt_history"],
        st.session_state["chat_answers_history"],
    ):
        message(user_query, is_user=True)
        message(generated_response, is_user=False)
