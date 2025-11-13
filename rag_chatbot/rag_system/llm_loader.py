"""
LLM 로딩 및 텍스트 생성 모듈
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from typing import Optional, Dict, Any

from .config import Config


class LLMLoader:
    """LLM 로더 클래스"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_quantization: Optional[bool] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            model_name: 모델 이름 (None이면 Config에서 가져옴)
            use_quantization: 양자화 사용 여부 (None이면 Config에서 가져옴)
            cache_dir: 모델 캐시 디렉토리
        """
        self.model_name = model_name or Config.LLM_MODEL_NAME
        self.use_quantization = use_quantization if use_quantization is not None else Config.LLM_QUANTIZATION
        
        if cache_dir is None:
            cache_dir = str(Config.MODEL_CACHE_DIR)
        self.cache_dir = cache_dir
        
        self.tokenizer = None
        self.model = None
        self.pipe = None
        
        self._load_model()
    
    def _load_model(self):
        """모델 로드"""
        print(f"모델 로딩 중: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        if self.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                quantization_config=quantization_config,
                device_map="auto",
                cache_dir=self.cache_dir
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
                cache_dir=self.cache_dir
            )
        
        self.model.eval()
        
        # Pipeline 생성
        self.pipe = pipeline(
            model=self.model,
            task='text-generation',
            tokenizer=self.tokenizer
        )
        
        print("모델 로딩 완료")
    
    def generate_answer(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        프롬프트에 대한 답변 생성
        
        Args:
            prompt: 사용자 프롬프트
            max_new_tokens: 최대 생성 토큰 수
            system_prompt: 시스템 프롬프트
        
        Returns:
            생성된 답변
        """
        if max_new_tokens is None:
            max_new_tokens = Config.LLM_MAX_NEW_TOKENS
        
        if system_prompt is None:
            system_prompt = "당신은 주어진 정보만을 사용하여 정확하게 답변하는 전문가입니다. 정보에 없는 내용은 추측하지 않습니다."
        
        # 메시지 구성
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # 템플릿 적용
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        # 텍스트 생성
        result = self.pipe(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        
        # 답변 추출
        answer = self._extract_answer(result[0]['generated_text'])
        
        return answer
    
    def _extract_answer(self, response: str) -> str:
        """생성된 텍스트에서 답변 부분만 추출"""
        try:
            # assistant 부분 추출
            if '<|im_start|>assistant' in response:
                response = response.split('<|im_start|>assistant')[1]
            
            # <think> 이후 부분 추출
            if '</think>' in response:
                think_end = response.find('</think>') + len('</think>')
                response = response[think_end:]
            
            response = response.strip()
            return response
        except Exception as e:
            print(f"답변 추출 중 오류: {e}")
            return response
    
    def create_prompt(
        self,
        query: str,
        context: str
    ) -> str:
        """
        RAG용 프롬프트 생성
        
        Args:
            query: 질문
            context: 검색된 컨텍스트
        
        Returns:
            포맷팅된 프롬프트
        """
        prompt = f"""아래의 <정보>만을 사용하여 <질문>에 답변하세요. 정보에 없는 내용은 추측하지 마세요.

<정보>
{context}

<질문>
{query}

<답변>
위 <정보>에서 질문과 관련된 내용만 찾아서 정확하게 답변하세요. 정보에 정확한 답이 없으면 "제공된 정보에서 해당 내용을 찾을 수 없습니다."라고 답변하세요.\n"""
        return prompt

