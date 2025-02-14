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

# ✅ Gemini API 키 설정
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ✅ 임베딩 모델 캐싱 (메모리 절약)
@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # Google의 임베딩 모델
        google_api_key=st.secrets["GOOGLE_API_KEY"]  # 기존에 설정한 API 키 사용
    )

# ✅ Google Drive API 초기화
@st.cache_resource
def init_drive_service():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["google_credentials"],
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    return build('drive', 'v3', credentials=credentials)

# ✅ PDF 파일 목록 가져오기
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

# ✅ Streamlit UI 시작
st.title("📄 IPR실 매뉴얼 AI 챗봇")
st.write("☆ 자료 수정 또는 추가 희망시 주영 연구원 연락 ☆")

try:
    # ✅ Google Drive에서 PDF 목록 가져오기
    service = init_drive_service()
    FOLDER_ID = '1fThzSsDTeZA6Zs1VLGNPp6PejJJVydra'
    pdf_files = get_pdf_files(service, FOLDER_ID)

    if not pdf_files:
        st.warning("📂 매뉴얼 폴더에 PDF 파일이 없습니다.")
    else:
        st.info(f"📄 총 {len(pdf_files)}개의 매뉴얼을 분석 중...")

        # ✅ 모든 PDF 파일 처리 및 임베딩 벡터 저장
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

                    os.unlink(pdf_path)  # 사용 후 파일 삭제

                except Exception as e:
                    st.warning(f"⚠️ {pdf['name']} 처리 중 오류 발생: {str(e)}")

            # ✅ 문서 분할 최적화
            text_splitter = ecursiveCharacterTextSplitter(
                chunk_size=1000,  
                chunk_overlap=100,  
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
                is_separator_regex=False
            )
            split_texts = text_splitter.split_documents(all_texts)

            # ✅ 벡터 저장소 생성 및 캐싱
            embeddings = get_embeddings()
            vector_store = FAISS.from_documents(split_texts, embeddings)

            return vector_store

        # ✅ 벡터 스토어 생성
        vector_store = process_all_pdfs()
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # 검색 결과 최적화

        # ✅ AI 프롬프트 설정
        system_template = """
        다음 문맥을 참고하여 사용자의 질문에 간단명료하게 답변하세요.
        답을 모르는 경우 "모르겠습니다"라고 답변하고, 답변을 만들어내려 하지 마세요.
        가능한 경우 정보의 출처(문서)를 언급해주세요.
        ----------------
        {context}
        """
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
        prompt = ChatPromptTemplate.from_messages(messages)

        # ✅ 메모리 설정
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",  # 출력 키를 명시적으로 지정
            verbose=False  # 추가
        )

        # ✅ LLM 모델 설정
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-8b",  # 수정된 모델명
            temperature=0.3,
            max_output_tokens=2048,
        )


        # ✅ 대화 체인 설정
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={'prompt': prompt},
            return_source_documents=True
        )

        # ✅ 채팅 인터페이스
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
