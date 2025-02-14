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

# PDF 처리 및 벡터 저장소 생성 (캐시 제거)
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

def main():
    st.title("📄 IPR실 매뉴얼 AI 챗봇")
    st.write("☆ 자료 수정 또는 추가 희망시 주영 연구원 연락 ☆")

    try:
        # Google Drive 연결 및 PDF 파일 가져오기
        service = init_drive_service()
        if not service:
            st.error("Google Drive 서비스 연결 실패")
            return

        folder_id = st.secrets["FOLDER_ID"]
        pdf_files = get_pdf_files(service, folder_id)

        if not pdf_files:
            st.warning("📂 매뉴얼 폴더에 PDF 파일이 없습니다.")
            return
        
        # 상태 표시를 위한 placeholder
        status_placeholder = st.empty()
        
        # PDF 처리 및 벡터 저장소 생성
        all_texts = []
        total_steps = len(pdf_files) + 2
        
        # PDF 파일 처리
        for idx, pdf in enumerate(pdf_files):
            progress = (idx / total_steps) * 100
            status_placeholder.info(f"📄 매뉴얼 분석 중... ({progress:.1f}%)\n\n현재 처리 중: {pdf['name']}")
            documents = process_pdf(pdf, service)
            all_texts.extend(documents)
        
        # 텍스트 분할
        progress = (len(pdf_files) / total_steps) * 100
        status_placeholder.info(f"📄 매뉴얼 분석 중... ({progress:.1f}%)\n\n텍스트 분할 작업 진행 중...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            is_separator_regex=False
        )
        split_texts = text_splitter.split_documents(all_texts)
        
        # 벡터 저장소 생성
        progress = ((len(pdf_files) + 1) / total_steps) * 100
        status_placeholder.info(f"📄 매뉴얼 분석 중... ({progress:.1f}%)\n\n벡터 저장소 생성 중...")
        
        embeddings = get_embeddings()
        vector_store = FAISS.from_documents(split_texts, embeddings)
        
        # 분석 완료 메시지
        status_placeholder.success("✅ 매뉴얼 분석이 완료되었습니다. 질문해 주세요!")

        # 나머지 코드는 동일하게 유지
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        system_template = """
        You are an expert AI assistant for IPR manuals. Base your answers strictly on the provided context.

        Guidelines:
        1. ALWAYS answer in Korean
        2. Use Markdown format
        3. Keep responses concise (2-4 sentences)
        4. If unsure, say "확실하지 않습니다"
        5. Cite source documents when possible
        Context:
        ----------------
        {context}
        """
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
        prompt = ChatPromptTemplate.from_messages(messages)

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            verbose=False
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            max_output_tokens=2048,
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={'prompt': prompt},
            return_source_documents=True
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

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
                            
                    st.session_state.messages.append({"role": "assistant", "content": response['answer']})

    except Exception as e:
        st.error(f"🚨 시스템 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
