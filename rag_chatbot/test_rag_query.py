"""
RAG 시스템 질의 테스트
데이터가 적재된 후 실행하세요.
"""
from rag_system.rag_pipeline import RAGPipeline

def main():
    print("=" * 70)
    print("RAG 시스템 질의 테스트")
    print("=" * 70)
    
    # RAG 파이프라인 초기화 (LLM 없이 먼저 테스트)
    print("\n[1단계] RAG 파이프라인 초기화 (LLM 제외)...")
    try:
        rag = RAGPipeline(
            collection_name="security_info",
            use_reranker=False,  # 먼저 reranker 없이 테스트
            load_llm=False  # LLM 없이 검색만 테스트
        )
        print("  [성공] RAG 파이프라인 초기화 완료")
    except Exception as e:
        print(f"  [오류] 초기화 실패: {e}")
        print("\n먼저 다음 명령을 실행하세요:")
        print("  python -m rag_system.ingest")
        return
    
    # 검색 테스트
    print("\n[2단계] 검색 테스트...")
    test_queries = [
        "2025년 1학기 수강신청 기간",
        "겨울 계절학기",
        "졸업사정",
        "복학 신청"
    ]
    
    for query in test_queries:
        print(f"\n질문: {query}")
        print("-" * 70)
        try:
            result = rag.query(query, top_k=3, use_llm=False)
            print(f"검색된 컨텍스트:")
            print(result["context"][:500] + "..." if len(result["context"]) > 500 else result["context"])
        except Exception as e:
            print(f"  [오류] {e}")
    
    print("\n" + "=" * 70)
    print("검색 테스트 완료!")
    print("\nLLM을 사용한 답변 생성을 원하면:")
    print("  rag = RAGPipeline(load_llm=True)")
    print("  result = rag.query('질문', use_llm=True)")

if __name__ == "__main__":
    main()

