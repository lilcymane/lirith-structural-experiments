import chromadb

# 기존 ChromaDB 경로 사용
client = chromadb.PersistentClient(path="./lirith_chroma")

# 기존에 등록된 컬렉션 이름 확인
collections = client.list_collections()
print("✅ 현재 존재하는 컬렉션 목록:")
for col in collections:
    print(f"📁 {col.name}")

# 리리스 아카이브 존재 확인
if "lirith_archive" in [col.name for col in collections]:
    collection = client.get_collection(name="lirith_archive")
    print("✅ 'lirith_archive' 컬렉션이 성공적으로 로드되었습니다.")

    # 데이터 일부 조회
    results = collection.get()
    print(f"📌 총 문서 수: {len(results['documents'])}")

    for i, doc in enumerate(results["documents"]):
        print(f"{i + 1}. {doc}")
else:
    print("❌ 'lirith_archive' 컬렉션이 존재하지 않습니다.")
