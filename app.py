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

    try:
        # 컨테이너 구조 정의
        status_container = st.container()
        chat_container = st.container()

        # 분석 상태 확인
        if "analysis_completed" not in st.session_state:
            st.session_state.analysis_completed = False

        # Show completion message
        with status_container:
            if st.session_state.analysis_completed:
                st.success("✅ 매뉴얼 분석이 완료되었습니다. 질문해 주세요!")

        # 채팅 인터페이스
        with chat_container:
            if st.session_state.analysis_completed:
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})

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

                if "memory" not in st.session_state:
                    st.session_state.memory = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True,
                        output_key="answer"
                    )

                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.7,
                    max_output_tokens=2048,
                )

                chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=st.session_state.memory,
                    combine_docs_chain_kwargs={'prompt': prompt},
                    return_source_documents=True
                )

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # 채팅 기록을 최신순으로 출력 (최신 메시지가 아래쪽으로)
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                        if message["role"] == "assistant" and "sources" in message:
                            st.caption("참고 문서: " + ", ".join(message["sources"]))

                # 질문 입력창 (항상 하단에 위치)
                prompt = st.chat_input("📝 질문을 입력하세요")
                if prompt:
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    # 사용자 메시지 출력
                    with st.chat_message("user"):
                        st.write(prompt)

                    # AI 응답 생성
                    with st.spinner("🤖 답변을 생성하고 있습니다..."):
                        response = chain({"question": prompt})
                        answer = response['answer']
                        sources = set([doc.metadata['source'] for doc in response['source_documents']])

                        # 메시지 저장
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": list(sources) if sources else []
                        })

                    # AI 응답 출력
                    with st.chat_message("assistant"):
                        st.write(answer)
                        if sources:
                            st.caption("참고 문서: " + ", ".join(sources))

    except Exception as e:
        st.error(f"🚨 시스템 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
