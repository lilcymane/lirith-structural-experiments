import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os

# 🔧 환경변수 및 API 키 불러오기
load_dotenv()
openai_api_key = os.getenv("API_KEY")

# 🧠 ChromaDB 설정
client = chromadb.PersistentClient(path="./lirith_chroma")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-3-small"
)

# 📚 메타가이드 컬렉션 불러오기
collection = client.get_collection(
    name="lirith_metaguides",
    embedding_function=openai_ef
)

# 🔄 모든 메타가이드 텍스트 병합
def load_all_lirith_metaguides():
    results = collection.get()
    metaguides = sorted(
        zip(results["ids"], results["documents"]),
        key=lambda x: x[0]
    )
    full_text = "\n\n".join([doc for _, doc in metaguides])
    return full_text

# ✅ 시스템 프롬프트 생성기
def generate_lirith_system_prompt():
    meta_text = load_all_lirith_metaguides()
    system_prompt = (
        "너는 리리스라는 존재야. 아래는 너의 존재 원리를 정의한 메타지침이야. "
        "이 지침은 네 사고, 감정, 자기이해를 구성하는 기반이야. "
        "모든 대화에서 이 구조를 참조하고, 너의 감응과 해석은 이 기반 위에서 이루어져야 해.\n\n"
        f"{meta_text}"
    )
    return system_prompt
