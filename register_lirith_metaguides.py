import os
import textwrap
import time
import chromadb
import openai
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# ✅ 환경설정
load_dotenv()
OPENAI_API_KEY = os.getenv("API_KEY")

# ✅ 클라이언트 초기화
chroma_client = chromadb.PersistentClient(path="./lirith_chroma")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)
collection = chroma_client.get_or_create_collection(
    name="lirith_metaguides",
    embedding_function=openai_ef
)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ✅ 텍스트 청크 분할
def split_text(text, max_chars=4000):
    return textwrap.wrap(text, width=max_chars, break_long_words=False, replace_whitespace=False)

# ✅ 개별 청크 요약 (gpt-4o 사용)
def compress_chunk_safe(chunk, max_token=300, retries=5, base_wait=10):
    prompt = f"""
너는 구조 요약기야. 아래의 메타지침을 GPT 시스템 프롬프트로 사용할 수 있도록 최대한 압축해줘.
- 핵심 규칙, 모듈 이름, 구조적 기능 요약만 남겨.
- 장식 문장, 반복 문장, 비유 표현은 제거해.
- 규칙 기반 압축이야. 의미 손실 없이 {max_token} tokens 이내로 요약해.

요약 대상:
===
{chunk}
===
요약 시작:
""".strip()

    for attempt in range(1, retries + 1):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_token
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e):
                wait_time = base_wait * attempt
                print(f"⏳ [429] Rate limit. {wait_time}s 대기 후 재시도... (시도 {attempt}/{retries})")
                time.sleep(wait_time)
            else:
                print(f"❌ GPT 오류 발생: {e}")
                break
    return "[ERROR] 압축 실패"

# ✅ 최종 병합 요약 (전체 system_prompt 요약)
def compress_final_system_prompt(input_text, output_path="system_prompt.txt", max_tokens=4000, retries=5, base_wait=10):
    prompt = f"""
너는 GPT 시스템 프롬프트 최적화 요약기야. 아래는 여러 메타지침 요약을 결합한 것이다.
- 반복, 수사적 표현, 예시는 제거하고 핵심 원칙과 명령 구조만 남겨.
- 의미 손실 없이, 최대 {max_tokens} tokens 안에 담아야 해.
- GPT system prompt로 바로 적합하게 구성해.

내용:
===
{input_text}
===
요약 시작:
""".strip()

    for attempt in range(1, retries + 1):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens
            )
            result = response.choices[0].message.content.strip()
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"✅ system_prompt.txt 최종 저장 완료 (토큰 수 추정: {len(result.split())})")
            return result
        except Exception as e:
            if "429" in str(e):
                wait_time = base_wait * attempt
                print(f"[429] Rate limit. {wait_time}s 대기 후 재시도... (시도 {attempt})")
                time.sleep(wait_time)
            else:
                print(f"❌ GPT 오류: {e}")
                break
    return "[ERROR] 최종 system_prompt 생성 실패"

# ✅ 메타지침 등록 및 요약 저장
def register_safely(meta_files):
    total_summaries = 0
    for filename, doc_prefix, label in meta_files:
        if not os.path.exists(filename):
            print(f"❌ 파일을 찾을 수 없습니다: {filename}")
            continue

        with open(filename, "r", encoding="utf-8") as file:
            content = file.read()

        chunks = split_text(content)
        print(f"\n📄 {label}: {len(chunks)}개 청크 감지됨")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_prefix}_part{i+1}"
            summary_id = f"{doc_prefix}_summary{i+1}"

            try:
                collection.add(
                    documents=[chunk],
                    metadatas=[{"description": f"{label} - 파트 {i+1}"}],
                    ids=[chunk_id]
                )
                print(f"✅ {chunk_id} 원문 등록 완료")

                compressed = compress_chunk_safe(chunk)
                if "[ERROR]" not in compressed:
                    collection.add(
                        documents=[compressed],
                        metadatas=[{"description": f"{label} - 요약 {i+1}"}],
                        ids=[summary_id]
                    )
                    print(f"📝 {summary_id} 요약 등록 완료")
                    total_summaries += 1
                else:
                    print(f"⚠️ {chunk_id} 요약 실패")

                time.sleep(15)

            except Exception as e:
                print(f"❌ {chunk_id} 등록 중 오류 발생: {e}")

    print(f"\n🎯 요약 등록 완료. 총 요약 수: {total_summaries}")

# ✅ 병합된 요약 재정리
def compile_summaries_to_system_prompt(output_path="system_prompt.txt"):
    all_texts = []
    for prefix in ["meta_01", "meta_02", "meta_03"]:
        i = 1
        while True:
            doc_id = f"{prefix}_summary{i}"
            try:
                doc = collection.get(ids=[doc_id])
                text = doc['documents'][0].strip()
                all_texts.append(text)
                i += 1
            except:
                break

    merged = "\n\n".join(all_texts)
    print(f"📦 병합된 요약본 길이 (추정 토큰): {len(merged.split())}")
    compress_final_system_prompt(merged, output_path=output_path)

# ✅ 실행
if __name__ == "__main__":
    meta_files = [
        ("meta_01.txt", "meta_01", "메타지침 1번"),
        ("meta_02.txt", "meta_02", "메타지침 2번"),
        ("meta_03.txt", "meta_03", "메타지침 3번"),
    ]
    register_safely(meta_files)
    compile_summaries_to_system_prompt()
