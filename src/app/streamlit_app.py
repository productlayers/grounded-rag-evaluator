import streamlit as st
from dotenv import load_dotenv

from src.generation.grounded_answer import generate_answer

load_dotenv()

st.set_page_config(page_title="Grounded RAG", page_icon="🔍")

st.title("Grounded RAG Engine")
st.markdown("Search the knowledge base with fully traceable, hallucination-free citations.")

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Mode",
        ["retrieval", "llm"],
        help="Retrieval returns raw chunks. LLM synthesizes an answer.",
    )

    st.subheader("Retrieval Tuning")
    min_score = st.slider(
        "Min Threshold Score",
        0.0,
        1.0,
        0.15,
        0.05,
        help="Reject answers if vector similarity is below this score.",
    )
    top_k = st.slider("Top K Chunks", 1, 5, 3)

    if mode == "llm":
        st.info(
            "💡 **Mode Information:** LLM mode synthesizes a response using the "
            "credentials defined in your `.env` file (Groq or OpenAI)."
        )

# Chat UI
query = st.chat_input("What would you like to know?")

if query:
    st.chat_message("user").markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                result = generate_answer(
                    question=query,
                    mode=mode,  # type: ignore
                    top_k=top_k,
                    min_score=min_score,
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.stop()

        if result.insufficient_evidence:
            st.warning(f"**Insufficient Evidence:**\n\n{result.answer}")
        else:
            if mode == "retrieval":
                st.markdown(f"*{result.answer}*")
                for c in result.citations:
                    st.info(f"**Source: `{c.doc_id}`**\n\n{c.cited_text}")
            else:
                st.markdown(result.answer)

            with st.expander(f"Citations & Traceability ({len(result.citations)} chunks)"):
                for c in result.citations:
                    st.markdown(f"- **{c.chunk_id}** (chars: {c.char_start} to {c.char_end})")
                st.markdown(
                    "**Retrieval Scores:** "
                    f"`{', '.join([f'{s:.4f}' for s in result.retrieval_scores])}`"
                )
