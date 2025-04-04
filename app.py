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

# 📌 환경 변수 설정
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

@st.cache_resource
def init_drive_service():
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["google_credentials"],
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        return build('drive', 'v3', credentials=credentials, cache_discovery=False)
    except Exception as e:
        st.error(f"Drive 서비스 초기화 오류: {str(e)}")
        return None

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

def process_pdf(pdf, service):
    try:
        request = service.files().get_media(fileId=pdf['id'])
        file_content = request.execute()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_content)
            pdf_path = temp_file.name

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        for doc in documents:
            doc.metadata["source"] = pdf["name"]

        os.unlink(pdf_path)
        return documents
    except Exception as e:
        st.warning(f"⚠️ {pdf['name']} 처리 중 오류 발생: {str(e)}")
        return []

def create_vector_store(texts, embeddings):
    try:
        return FAISS.from_documents(texts, embeddings)
    except Exception as e:
        st.error(f"벡터 저장소 생성 중 오류 발생: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="기술 소개 챗봇", layout="wide")
    st.title("💡 보유 기술 안내 챗봇")
    st.write("우리 회사가 보유한 기술 중 궁금한 내용을 물어보세요.")

    try:
        service = init_drive_service()
        embeddings = get_embeddings()

        if not service or not embeddings:
            st.stop()

        folder_id = st.secrets.get("FOLDER_ID")
        if not folder_id:
            st.error("📂 폴더 ID가 설정되지 않았습니다.")
            return

        pdf_files = get_pdf_files(service, folder_id)
        if not pdf_files:
            st.warning("📂 PDF 문서가 없습니다.")
            return

        status_container = st.container()
        chat_container = st.container()

        if "analysis_completed" not in st.session_state:
            st.session_state.analysis_completed = False
            status_placeholder = status_container.empty()
            all_texts = []

            for idx, pdf in enumerate(pdf_files, 1):
                status_placeholder.info(f"📄 기술 문서 분석 중... ({idx}/{len(pdf_files)})\n현재: {pdf['name']}")
                documents = process_pdf(pdf, service)
                all_texts.extend(documents)

            status_placeholder.info("🧠 텍스트 분할 중...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            split_texts = text_splitter.split_documents(all_texts)

            status_placeholder.info("🧠 벡터 저장소 생성 중...")
            vector_store = create_vector_store(split_texts, embeddings)

            if not vector_store:
                return

            st.session_state.vector_store = vector_store
            st.session_state.all_pdfs = pdf_files
            st.session_state.analysis_completed = True

        if st.session_state.analysis_completed:
            with status_container:
                st.success("✅ 기술 자료 분석 완료! 궁금한 기술을 질문해보세요.")

            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
            system_template = """
            너는 우리 회사가 보유한 기술 관련 PDF 자료를 기반으로 질문자의 요청에 응답하는 AI 기술 컨설턴트입니다.

            다음 기준에 따라 질문에 답변하세요:

            1. 질문자가 어떤 기술에 관심이 있는지 파악하여, 그와 관련된 **우리 회사가 보유한 기술**을 안내합니다.
            2. 가능한 경우, 관련 기술이 언급된 **문서 이름과 페이지 번호**를 함께 명시합니다.
            3. 답변은 반드시 **한국어**로 작성하며, **2~4문장 이내**로 간결하게 설명합니다.
            4. 관련된 기술이 확실하지 않으면 "확실하지 않습니다"라고 말합니다.
            5. 모든 답변은 제공된 기술자료 내용에 기반해야 하며, 추측하지 않습니다.

            기술자료 요약:
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
                model="gemini-2.0-pro",
                temperature=0.7,
                max_output_tokens=2048,
            )

            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=st.session_state.memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                return_source_documents=True
            )

            if "messages" not in st.session_state:
                st.session_state.messages = []

            # 📝 사용자 입력 처리
            if prompt := st.chat_input("궁금한 기술이나 특허에 대해 질문해보세요"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.spinner("🤖 답변 생성 중..."):
                    response = chain({"question": prompt})
                    answer = response["answer"]

                    # 🔍 출처 문서 정리
                    source_docs = response.get("source_documents", [])
                    sources = set()
                    for doc in source_docs:
                        filename = doc.metadata.get("source", "알 수 없는 문서")
                        page = doc.metadata.get("page", "알 수 없는 페이지")
                        sources.add(f"- `{filename}`, 페이지 {page}")

                    # 🔧 답변에 출처 추가
                    if sources:
                        answer += "\n\n---\n📄 **참고 문서:**\n" + "\n".join(sources)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

            # 💬 채팅 메시지 출력
            for i in range(len(st.session_state.messages) - 1, -1, -2):
                if i > 0 and st.session_state.messages[i - 1]["role"] == "user":
                    st.markdown(f"**🙋 질문:** {st.session_state.messages[i - 1]['content']}")
                st.markdown(f"**🤖 답변:** {st.session_state.messages[i]['content']}")

            # 📚 PDF 미리보기
            with st.expander("📂 PDF 문서 미리보기", expanded=False):
                for pdf in st.session_state.all_pdfs:
                    file_id = pdf["id"]
                    file_name = pdf["name"]
                    preview_url = f"https://drive.google.com/file/d/{file_id}/preview"
                    st.markdown(f"**📄 {file_name}**")
                    st.components.v1.iframe(preview_url, height=400)

    except Exception as e:
        st.error(f"🚨 시스템 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
