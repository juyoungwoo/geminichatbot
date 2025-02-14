import streamlit as st
import os
import io
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
   ChatPromptTemplate,
   SystemMessagePromptTemplate,
   HumanMessagePromptTemplate,
)
from google.oauth2 import service_account
from googleapiclient.discovery import build
import tempfile

st.write("🔍 Checking secrets...")

if "GOOGLE_API_KEY" not in st.secrets:
    st.error("🚨 `GOOGLE_API_KEY`가 `secrets.toml`에 설정되지 않았습니다.")
else:
    st.success("✅ `GOOGLE_API_KEY` 로드 성공!")

if "google_credentials" not in st.secrets:
    st.error("🚨 `google_credentials`가 `secrets.toml`에 설정되지 않았습니다.")
else:
    st.success("✅ `google_credentials` 로드 성공!")

# Google Gemini API 키 설정
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# 임베딩 모델 캐싱
@st.cache_resource
def get_embeddings():
   return GoogleGenerativeAIEmbeddings(model="text-embedding-004")

# Google Drive 설정
@st.cache_resource
def init_drive_service():
   credentials = service_account.Credentials.from_service_account_info(
       st.secrets["google_credentials"],
       scopes=['https://www.googleapis.com/auth/drive.readonly']
   )
   return build('drive', 'v3', credentials=credentials)

# PDF 파일 목록 가져오기
def get_pdf_files(service, folder_id):
   results = service.files().list(
       q=f"'{folder_id}' in parents and mimeType='application/pdf'",
       fields="files(id, name)"
   ).execute()
   return results.get('files', [])

# Streamlit UI 구성
st.title("📄 IPR실 매뉴얼 AI 챗봇")
st.write("☆ 자료 수정 또는 추가 희망시 주영 연구원 연락 ☆")

try:
   # 드라이브 서비스 초기화
   service = init_drive_service()
   FOLDER_ID = '1fThzSsDTeZA6Zs1VLGNPp6PejJJVydra'

   # PDF 파일 목록 가져오기
   pdf_files = get_pdf_files(service, FOLDER_ID)

   if not pdf_files:
       st.warning("폴더에 PDF 파일이 없습니다.")
   else:
       st.info(f"총 {len(pdf_files)}개의 매뉴얼을 분석합니다.")

       # 모든 PDF 파일의 내용을 하나로 합치기
       @st.cache_resource
       def process_all_pdfs():
           all_texts = []
           for pdf in pdf_files:
               try:
                   request = service.files().get_media(fileId=pdf['id'])
                   file_content = request.execute()

                   with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                       temp_file.write(file_content)
                       pdf_path = temp_file.name

                   loader = PyPDFLoader(pdf_path)
                   documents = loader.load()

                   for doc in documents:
                       doc.metadata['source'] = pdf['name']

                   all_texts.extend(documents)
                   os.unlink(pdf_path)

               except Exception as e:
                   st.warning(f"{pdf['name']} 처리 중 오류 발생: {str(e)}")

           # 문서 분할 최적화
           text_splitter = CharacterTextSplitter(
               chunk_size=500,  
               chunk_overlap=200,
               separator="\n"
           )
           split_texts = text_splitter.split_documents(all_texts)

           # 캐시된 임베딩 모델 사용
           embeddings = get_embeddings()
           vector_store = FAISS.from_documents(split_texts, embeddings)

           return vector_store

       # 모든 PDF 처리
       vector_store = process_all_pdfs()
       retriever = vector_store.as_retriever(search_kwargs={"k": 3})

       # 프롬프트 설정
       system_template = """
       Use the following pieces of context to answer the users question shortly.
       Given the following summaries of a long document and a question.
       If you don't know the answer, just say that "I don't know", don't try to make up an answer.
       If possible, mention which document (source) the information comes from.
       ----------------
       {summaries}
       You MUST answer in Korean and in Markdown format:
       """
       messages = [
           SystemMessagePromptTemplate.from_template(system_template),
           HumanMessagePromptTemplate.from_template("{question}")
       ]
       prompt = ChatPromptTemplate.from_messages(messages)
       chain_type_kwargs = {"prompt": prompt}

       # Google Gemini LLM 설정
       llm = ChatGoogleGenerativeAI(
           model="gemini-1.5-flash",
           temperature=0,
           google_api_key=os.environ["GOOGLE_API_KEY"],
           max_tokens=512
       )

       # Conversational QA 체인 설정
       chain = ConversationalRetrievalChain.from_llm(
           llm=llm,
           retriever=retriever,
           return_source_documents=True,
           chain_type_kwargs=chain_type_kwargs
       )

       # 대화 기록 저장
       if "chat_history" not in st.session_state:
           st.session_state.chat_history = []

       # 사용자 질문 입력
       query = st.text_input("질문을 입력하세요:")
       if st.button("질문하기") and query:
           with st.spinner("답변 생성 중..."):
               result = chain({"question": query, "chat_history": st.session_state.chat_history}, return_only_outputs=True)
               answer = result["answer"]
               st.session_state.chat_history.append((query, answer))
               st.markdown(answer)

except Exception as e:
   st.error(f"오류가 발생했습니다: {str(e)}")
