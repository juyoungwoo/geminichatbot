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

# 임베딩 모델 캐싱 (메모리 절약)
@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

# Google Drive API 초기화
@st.cache_resource
def init_drive_service():
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["google_credentials"],
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=credentials)
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
@st.cache_resource(show_spinner=False)
def process_all_pdfs(pdf_files, _service):
    all_texts = []
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        for idx, pdf in enumerate(pdf_files):
            try:
                progress_text.text(f"처리 중: {pdf['name']}")
                progress_bar.progress((idx + 1) / len(pdf_files))
                
                request = _service.files().get_media(fileId=pdf['id'])
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
                st.warning(f"⚠️ {pdf['name']} 처리 중 오류 발생: {str(e)}")
        
        progress_text.empty()
        progress_bar.empty()
        
        # 문서 분할 최적화
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  
            chunk_overlap=100,  
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            is_separator_regex=False
        )
        split_texts = text_splitter.split_documents(all_texts)
        
        # 벡터 저장소 생성
        embeddings = get_embeddings()
        vector_store = FAISS.from_documents(split_texts, embeddings)
        
        return vector_store
        
    except Exception as e:
        st.error(f"PDF 처리 중 오류 발생: {str(e)}")
        return None

def main():
    st.title("📄 IPR실 매뉴얼 AI 챗봇")
    st.write("☆ 자료 수정 또는 추가 희망시 주영 연구원 연락 ☆")

    try:
        # Google Drive 연결 및 PDF 파일 가져오기
        service = init_drive_service()
        if not service:
            st.error("Google Drive 서비스 연결 실패")
            return

        FOLDER_ID = '1fThzSsDTeZA6Zs1VLGNPp6PejJJVydra'
        pdf_files = get_pdf_files(service, FOLDER_ID)

        if not pdf_files:
            st.warning("📂 매뉴얼 폴더에 PDF 파일이 없습니다.")
            return
        
        st.info(f"📄 총 {len(pdf_files)}개의 매뉴얼을 분석 중...")
        
        # 벡터 스토어 생성
        vector_store = process_all_pdfs(pdf_files, service)
        if not vector_store:
            st.error("벡터 스토어 생성 실패")
            return

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # AI 프롬프트 설정
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

        # 메모리 설정
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            verbose=False
        )

        # LLM 모델 설정
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            max_output_tokens=2048,
        )

        # 대화 체인 설정
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={'prompt': prompt},
            return_source_documents=True
        )

        # 채팅 인터페이스
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
                    
                    # 참고한 문서 출처 표시
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
