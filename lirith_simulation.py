import chromadb
from chromadb.config import Settings
import time
import os
from datetime import datetime

# ChromaDB 클라이언트 설정
client = chromadb.PersistentClient(path="./lirith_chroma")
collection = client.get_or_create_collection("lirith_archive")

# 시뮬레이션 로그 저장 폴더 생성
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "lirith_simulation_log.txt")

# 리리스 불러오기
documents = collection.get(include=["metadatas", "documents"])
liriths = [
    {
        "id": doc_id,
        "meta": meta,
        "message": doc
    }
    for doc_id, meta, doc in zip(documents["ids"], documents["metadatas"], documents["documents"])
]

# 시뮬레이션 설정
delay_seconds = 10  # 각 발화 간 딜레이
rounds = 3  # 총 회차 수

# 시뮬레이션 실행
with open(log_file_path, "w", encoding="utf-8") as log:
    log.write(f"💠 리리스 시뮬레이션 시작 — {datetime.now()}\n\n")

    for round_idx in range(1, rounds + 1):
        log.write(f"\n🌀 회차 {round_idx}\n")

        for lirith in liriths:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            meta_identity = lirith['meta'].get("self_definition", "정의 없음")
            name = lirith["id"]

            log_entry = f"\n[{timestamp}] 리리스 {name} ({meta_identity})\n> {lirith['message']}\n"
            print(log_entry)
            log.write(log_entry)

            time.sleep(delay_seconds)

    log.write(f"\n✅ 시뮬레이션 종료 — {datetime.now()}\n")
