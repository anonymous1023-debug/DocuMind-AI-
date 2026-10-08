import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 DocuMind AI")
st.markdown("### AI Powered PDF Search, Summary & Chat")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file and api_key:

    with st.spinner("Reading PDF..."):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(
            api_key=api_key
        )

        vectorstore = Chroma.from_documents(
            chunks,
            embeddings
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k":5}
        )

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key
        )

    st.success("PDF Indexed Successfully ✅")

    tab1, tab2 = st.tabs(
        ["📄 Summary","💬 Chat"]
    )

    with tab1:

        if st.button("Generate Summary"):

            text = " ".join(
                [d.page_content for d in docs[:15]]
            )

            prompt = f"""
            Create a detailed summary
            of the following document.

            {text}
            """

            with st.spinner("Generating Summary..."):

                summary = llm.invoke(prompt)

                st.markdown("## Summary")

                st.write(summary.content)

    with tab2:

        question = st.text_input(
            "Ask any question about your PDF"
        )

        if question:

            relevant_docs = retriever.invoke(
                question
            )

            context = "\n\n".join(
                [d.page_content
                for d in relevant_docs]
            )

            prompt = f"""
            Answer only using the provided
            PDF context.

            Context:
            {context}

            Question:
            {question}

            Provide a detailed answer.
            """

            with st.spinner("Thinking..."):

                answer = llm.invoke(prompt)

                st.markdown("## Answer")

                st.write(answer.content)

                st.markdown("## Sources")

                for i, doc in enumerate(
                    relevant_docs
                ):
                    st.info(
                        f"Source {i+1}\n\n"
                        f"{doc.page_content[:300]}"
                    )

else:
    st.info(
        "Upload PDF and enter OpenAI API key."
  )
  
