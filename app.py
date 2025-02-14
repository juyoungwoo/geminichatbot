import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Gemini API 키 설정
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# 임베딩 모델 캐싱
@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

# Google Drive API 초기화 (개선된 버전)
@st.cache_resource
def init_drive_service():
    try:
        # 서비스 계정 인증 정보 확인
        if "google_credentials" not in st.secrets:
            st.error("Google 서비스 계정 인증 정보가 없습니다.")
            return None
            
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["google_credentials"],
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        
        service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
        return service
        
    except Exception as e:
        st.error(f"Drive 서비스 초기화 오류: {str(e)}")
        return None

# PDF 파일 목록 가져오기
def get_pdf_files(service, folder_id):
    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents and mimeType='application/pdf'",
            fields="files(id, name)"
        ).execute()
        return results.get('files', [])
    except Exception as e:
        st.error(f"Google Drive API 오류: {str(e)}")
        return []

# PDF 처리 및 벡터 저장소 생성
def process_pdf(pdf, service):
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
        
        os.unlink(pdf_path)
        return documents
    except Exception as e:
        st.warning(f"⚠️ {pdf['name']} 처리 중 오류 발생: {str(e)}")
        return []

# 벡터 저장소 생성 함수 추가
def create_vector_store(texts, embeddings):
    try:
        return FAISS.from_documents(texts, embeddings)
    except Exception as e:
        st.error(f"벡터 저장소 생성 중 오류 발생: {str(e)}")
        return None

def main():
    st.title("📄 IPR실 매뉴얼 AI 챗봇")
    st.write("☆ 자료 수정 또는 추가 희망시 주영 연구원 연락 ☆")

    try:  # 여기서 try 블록 시작
        # Initialize services
        service = init_drive_service()
        embeddings = get_embeddings()
        
        if not service or not embeddings:
            st.error("필수 서비스 초기화 실패")
            return

        folder_id = st.secrets.get("FOLDER_ID")
        if not folder_id:
            st.error("폴더 ID가 설정되지 않았습니다.")
            return

        pdf_files = get_pdf_files(service, folder_id)
        if not pdf_files:
            st.warning("📂 매뉴얼 폴더에 PDF 파일이 없습니다.")
            return

        status_container = st.container()
        chat_container = st.container()
        
        # ...나머지 코드...

        with chat_container:
            if st.session_state.analysis_completed:
                # 초기화 코드...

                # Handle new messages
                if prompt := st.chat_input("📝 질문을 입력하세요"):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("🤖 답변을 생성하고 있습니다..."):
                            response = chain({"question": prompt})
                            st.markdown(response['answer'])
                            
                            sources = set([doc.metadata['source'] for doc in response['source_documents']])
                            if sources:
                                st.markdown("---")
                                st.markdown("**참고한 문서:**")
                                for source in sources:
                                    st.markdown(f"- {source}")
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response['answer']
                            })

                # Display chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

    except Exception as e:  # 여기서 try 블록 끝
        st.error(f"🚨 시스템 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
        
if __name__ == "__main__":
    main()
