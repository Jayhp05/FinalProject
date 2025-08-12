import os
import streamlit as st
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
import streamlit.components.v1 as components

OLLAMA_API_URL = "http://127.0.0.1:11434" 

# 문서 예시 - 스포츠 경매 데이터 예시 (여기선 임시로 설정)
sports_auction_info = [
    {
        "경매번호": "1",
        "종목": "축구",
        "판매자": "관리자",
        "경매시작가": "1만원",
        "마감일": "2025-07-20",
        "특이사항": "없음"
    },
]

# 문서화
documents = [
    Document(
        page_content=", ".join([
            f"{key}: {value}" for key, value in item.items()
        ])
    )
    for item in sports_auction_info
]

# 임베딩 & 벡터DB 생성
embedding_function = SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")
db = FAISS.from_documents(documents, embedding_function)

retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 5})

# Ollama 모델 설정
llm = ChatOllama(model="gemma3", temperature=0.5, base_url=OLLAMA_API_URL)

template = """
너는 'DropWin'이라는 경매 플랫폼에서 활동하는 친절하고 똑똑한 챗봇이야.

대답은 항상 존댓말로, 간단명료하고 친절하게 안내해줘.

<사용자 질문에 대한 참고 정보> 
{context}

질문: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# 체인 구성
chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | llm

# --- Streamlit UI 시작 ---
st.title("🏅 스포츠 경매 챗봇")
st.write("DropWin에 대해 궁금한 점을 물어보세요!")

st.markdown("""
<style>
    .chat-user {
        background-color:#D0E8FF; 
        padding:10px; 
        border-radius:15px; 
        margin:10px 0; 
        max-width:80%; 
        float:right; 
        clear:both;
    }
    .chat-bot {
        background-color:#F1F0F0; 
        padding:10px; 
        border-radius:15px; 
        margin:10px 0; 
        max-width:80%; 
        float:left; 
        clear:both;
    }
</style>
""", unsafe_allow_html=True)

# 채팅 히스토리 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 새 채팅 버튼
if st.button("🔄 새 채팅 시작"):
    st.session_state.chat_history = []
    st.rerun()

def render_message(speaker, message):
    if speaker == "🙂 사용자":
        st.markdown(f'<div class="chat-user"><b>{speaker}:</b> {message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bot"><b>{speaker}:</b> {message}</div>', unsafe_allow_html=True)

# 세션 상태 초기화 부분은 UI 위에 미리 선언되어 있어야 합니다
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 대화 내용")
    for speaker, message in st.session_state.chat_history:
        render_message(speaker, message)

with st.form("question_form", clear_on_submit=True):
    cols = st.columns([8,1])
    user_input = cols[0].text_input("질문을 입력하세요:")
    submit = cols[1].form_submit_button("전송")

if submit and user_input:
    response = chain.invoke({'question': user_input}).content
    st.session_state.chat_history.append(("🙂 사용자", user_input))
    st.session_state.chat_history.append(("🤖 챗봇", response))
    st.experimental_rerun()