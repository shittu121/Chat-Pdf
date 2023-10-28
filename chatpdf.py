import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import faiss
import pickle
from dotenv import load_dotenv
import os

# sidebar contents
with st.sidebar:
    st.title("Chat Pdf App")
    st.markdown('''
    ## About
    This App is an LLM-powered chatbot

    ''')
    add_vertical_space(3)
    st.write("Made with by faruq")

load_dotenv()
def main():
    st.header("Chat With PDF")

    # upload a pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
      pdf_reader = PdfReader(pdf)

      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()

      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=1000,
          chunk_overlap=200,
          length_function=len
         )
      chunks = text_splitter.split_text(text=text)
      
      # embeddings
     
      store_name = pdf.name[:-4]

      if os.path.exists(f"{store_name}.pk1"):
         with open(f"{store_name}.pk1", "rb") as f:
             VectorStore = pickle.load(f)
         #st.write('Embeddings loaded from the Disk')
      else:
           embeddings = OpenAIEmbeddings()
           VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
           with open(f"{store_name}.pk1", "wb") as f:
                pickle.dump(VectorStore, f)

      # Accept user Question
      query = st.text_input("Ask question about your pdf")
      #st.write(query)

      if query:
        docs = VectorStore.similarity_search(query=query, k=3)

        llm = OpenAI()
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        st.write(response)

        #st.write(docs)
           #st.write('Embeddings Compution Completed')
      # st.write(chunks)


if __name__ == '__main__':
    main()