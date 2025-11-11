"""
RAG 시스템 대화형 인터페이스
터미널에서 RAG 시스템과 대화할 수 있습니다.
"""
from rag_system.rag_pipeline import RAGPipeline

def main():
    print("=" * 70)
    print("RAG 시스템 대화형 인터페이스")
    print("=" * 70)
    print("\n[주의] LLM 로드는 시간이 걸릴 수 있습니다 (몇 분 소요)")
    print("모델 다운로드가 필요하면 더 오래 걸릴 수 있습니다.\n")
    
    # LLM 사용 여부 선택
    use_llm_input = input("LLM을 사용하시겠습니까? (y/n, 기본값: n): ").strip().lower()
    use_llm = use_llm_input == 'y' or use_llm_input == 'yes'
    
    # RAG 파이프라인 초기화
    print("\n[초기화 중...]")
    try:
        rag = RAGPipeline(
            collection_name="security_info",
            use_reranker=False,
            load_llm=use_llm
        )
        print("[초기화 완료!]\n")
    except Exception as e:
        print(f"[오류] 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("=" * 70)
    print("질문을 입력하세요. 종료하려면 'quit', 'exit', 'q'를 입력하세요.")
    print("=" * 70)
    print()
    
    # 대화 루프
    while True:
        try:
            # 질문 입력
            question = input("질문: ").strip()
            
            # 종료 명령 확인
            if question.lower() in ['quit', 'exit', 'q', '종료']:
                print("\n대화를 종료합니다.")
                break
            
            if not question:
                print("질문을 입력해주세요.\n")
                continue
            
            # 질문 처리
            print("\n[검색 중...]")
            result = rag.query(question, top_k=3, use_llm=use_llm)
            
            # 결과 출력
            print("\n" + "-" * 70)
            print("[검색된 정보]")
            print("-" * 70)
            context = result.get("context", "")
            if context and context != "관련 문서를 찾을 수 없습니다.":
                # 처음 500자만 표시
                if len(context) > 500:
                    print(context[:500] + "...")
                else:
                    print(context)
            else:
                print("관련 정보를 찾을 수 없습니다.")
            
            # LLM 답변 출력
            if use_llm and result.get("answer"):
                print("\n" + "-" * 70)
                print("[생성된 답변]")
                print("-" * 70)
                print(result["answer"])
            
            print("\n" + "=" * 70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n대화를 종료합니다.")
            break
        except Exception as e:
            print(f"\n[오류] {e}")
            import traceback
            traceback.print_exc()
            print()

if __name__ == "__main__":
    main()

