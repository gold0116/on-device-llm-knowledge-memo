# 온디바이스 LLM(On‑Device LLM) 기술 개요

온디바이스 LLM은 클라우드 서버가 아니라 **스마트폰, PC, IoT 디바이스 내부에서 직접 LLM을 실행하는 기술**을 의미한다.  
기기 내부의 CPU/GPU/NPU를 사용해 **네트워크 연결 없이 즉시 추론**을 수행한다.

---

## 1. 온디바이스 LLM의 필요성

### 1.1 프라이버시 및 데이터 보안
- 데이터가 클라우드로 전송되지 않기 때문에 개인정보 유출 위험 감소  
- 민감 정보(메시지, 사진, 음성 등)를 로컬에서 처리 가능

### 1.2 지연(latency) 감소
- 네트워크 왕복 없이 즉시 응답  
- 음성 비서·카메라 AI·번역 등 실시간 응용에 최적

### 1.3 오프라인 동작
- 인터넷 환경이 불안정한 환경에서도 일관적 성능  
- 여행, 군사용, 산업 환경에서 유용

### 1.4 비용 감소
- 서버 비용, GPU 인프라 유지비 감소  
- 대량 사용자 기반 서비스에서 비용 효과 큼

---

## 2. 온디바이스 LLM을 가능하게 만드는 핵심 기술

온디바이스는 리소스가 매우 제한적이므로 **경량화 기술**이 필수이다.

### 2.1 양자화(Quantization)
모델의 가중치를 FP16 → INT8 → INT4로 줄여  
- 모델 사이즈 축소  
- 메모리 사용 감소  
- 추론 속도 향상  

대표적 기법:
- GPTQ  
- AWQ  
- INT4/INT3 양자화  
- Apple MLX 4bit quantization  
- Qualcomm AI Engine 양자화 도구

### 2.2 지식 증류(Knowledge Distillation)
대형 모델(Teacher)의 지식을  
소형 모델(Student)로 압축 전달하는 기술.

효과:
- 파라미터 수를 10~50배 줄여도 성능 유지  
- 다양한 모바일 모델이 이 방식으로 훈련됨  
  (Phi-2, Gemini Nano, Gauss 등)

### 2.3 프루닝(Pruning)
- 중요도가 낮은 노드를 제거하여 모델을 슬림화  
- 구조적 프루닝(Structured)·비구조적(Unstructured) 모두 활용

### 2.4 경량 아키텍처 설계
온디바이스 모델은 대형 모델과 다르게 처음부터 경량 구조로 설계된다.

예:
- Google Gemini Nano  
- Microsoft Phi 시리즈  
- Samsung Gauss Nano  
- Mamba/SSM 기반 구조  
- Llama 3.2 “Instruct 1B/3B”  
- Qwen2.5 “Mini” 시리즈  

### 2.5 최적화된 추론 엔진
기기마다 최적 엔진이 다르다.

- **모바일(Android)**: Qualcomm AI Hub / SNPE / TFLite  
- **iOS/macOS**: MLX, CoreML  
- **PC CPU/GPU**: GGUF + llama.cpp  
- **NVIDIA GPU**: TensorRT‑LLM  
- **Chrome/Edge 브라우저**: WebGPU 기반 실행  

---

## 3. sLLM(Small LLM)의 부상

### 3.1 글로벌 기업의 전략 변화
- Google: Gemini Nano (멀티모달, 초경량)  
- Microsoft: Phi·Orca 시리즈 (증류 기반 고성능 sLLM)  
- Meta: Llama 3.2 3B/1B (초경량 범용 모델)  
- Samsung: Gauss(가우스) Nano → Galaxy S24에 적용  
- Apple: MM1 기반 iOS 온디바이스 모델 개발 중  

### 3.2 왜 sLLM이 중요해졌는가?
- 스마트폰에서 LLM을 직접 실행하는 수요 증가  
- 프라이버시 강화 규제  
- 클라우드 비용 증가  
- 멀티모달 모델의 로컬 적용 확대  

### 3.3 용도
- 메시지 요약  
- 카메라 씬 인식  
- 음성비서·명령 인식  
- 앱 자동화  
- 오프라인 번역  
- 로컬 RAG 기반 비서

---

## 4. 온디바이스 LLM의 한계

### 4.1 대규모 지식 처리의 어려움
- 파라미터 수가 작기 때문에 일반 지식량이 제한적  
- 해결책: **온디바이스 + 로컬 RAG** 조합

### 4.2 메모리 제약
- 1B~4B 모델이 실질적 한계 (모바일 기준)  
- 고성능 디바이스(iPhone Pro/고사양 PC)에서만 7B 가능

### 4.3 배터리 및 발열
- 추론 시 전력 소모 증가 가능  
- NPU 오프로딩 최적화 필요

---

## 5. 온디바이스 LLM 구축 시 권장 모델 목록 (2026 초 기준)

| 모델 | 크기 | 특징 | 실행 포맷 |
|------|------|--------|-----------|
| **Gemini Nano 1.6B/3B** | 초경량 | 멀티모달 지원 | TFLite, Android NPU |
| **Microsoft Phi‑2 / Phi‑3 Mini** | 2.7B | 고품질 증류 모델 | ONNX, GGUF |
| **Llama 3.2 1B/3B** | 1–3B | 범용, 품질 우수 | GGUF, ONNX |
| **Qwen2.5 Mini** | 1.8B | 한국어 성능 우수 | GGUF |
| **Mamba-based models** | 매우 경량 | SSM 기반 고효율 | PyTorch, MLX |
| **Gauss Nano** | 비공개 | Galaxy 온디바이스 | Samsung NPU |

---

## 6. 온디바이스 LLM 선택 기준

1. **메모리 사용량 (VRAM/RAM)**  
2. **응답 지연 시간**  
3. **한국어 성능 여부**  
4. **모바일/PC 실행 포맷 지원 여부**  
   - GGUF, ONNX, TFLite, CoreML, MLX  
5. **멀티모달(이미지/음성) 필요 여부**  
6. **프라이버시 요구 수준**  
7. **배터리 소모/발열**  

---

## 7. 결론

온디바이스 LLM은 지금부터 본격적인 대중화 단계에 진입하고 있으며,  
**경량화 기술(양자화·증류·프루닝) + 최적화된 추론 엔진 + sLLM 아키텍처**가 결합되며  
모바일/PC/IoT 환경에서 “실시간 AI 비서”와 “오프라인 AI”가 실현되고 있다.

본 문서는 Copilot 에이전트가 온디바이스 AI에 대한 전문적인 답변을 제공할 수 있도록  
핵심 개념·기술·모델 트렌드를 정리하였다.
