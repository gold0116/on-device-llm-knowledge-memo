## 📘 **Super Knowledge Page: 온디바이스 LLM · sLLM · 경량화 기술 통합 문서**

### # 1. 개요: 온디바이스 LLM이란 무엇인가?

온디바이스(On‑Device) LLM은 클라우드 서버가 아닌 **기기 내부(스마트폰·PC·IoT)에서 직접 추론하는 대규모 언어 모델 기술**을 의미한다.  
이 방식은 프라이버시 보호, 지연 시간 감소, 오프라인 기능, 비용 절감 등의 이점을 제공한다.  
온디바이스 AI는 실제로 **삼성 Galaxy S24**, **Google Pixel**, **iPhone Pro**, **Snapdragon/Xiaomi 디바이스** 등에서 널리 적용되고 있다.

***

# 2. 온디바이스 LLM이 필요한 이유 (정리)

### ✔ 2.1 프라이버시 강화

데이터가 외부 서버로 전송되지 않아 개인정보 유출 위험이 크게 줄어든다.

### ✔ 2.2 즉시 응답 (Low Latency)

네트워크 요청 없이 기기에서 바로 추론 → 빠른 응답.

### ✔ 2.3 오프라인 동작

인터넷 연결이 없는 환경에서도 고급 AI 성능 유지.

### ✔ 2.4 비용 절감

클라우드 GPU 비용이 없어지고, 대규모 사용자 기반 서비스에 적합.

***

# 3. 주요 웹사이트 요약 (URL 제한 때문에 문서에 편입하는 지식)

아래 웹사이트들은 Copilot 에이전트에 URL 4개 제한 때문에 직접 넣을 수 없는 핵심 자료들이다.  
그래서 이 문서 안에 **요약 형태로 포함**한다.

***

## 3.1 DeepMind Blog (요약)

DeepMind Blog는 Google DeepMind의 공식 연구 블로그로,

*   **최신 멀티모달 모델**,
*   **강화학습·기초 AI 알고리즘**,
*   **Gemini/Imagen 프로젝트 기술**,
*   **안전성·윤리 관련 분석**

등을 제공한다. 최신 AI 트렌드와 연구 경향을 이해하는 데 필수적이다.

***

## 3.2 TechCrunch – AI (요약)

TechCrunch AI 섹션은

*   최신 AI 스타트업
*   투자 라운드
*   기업용 AI 기술 도입
*   오픈소스 모델 개발 소식

을 빠르게 업데이트한다.  
특히 **경량화 모델**, **LLM API**, **Agent·RAG 스타트업** 관련 정보를 빠르게 얻을 수 있다.

***

## 3.3 VentureBeat AI (요약)

VentureBeat AI는 기업·엔터프라이즈 중심의 AI 동향을 제공한다.

*   기업용 LLM 도입 사례
*   AI 칩/하드웨어 동향
*   모델 최적화·추론 비용 절감
*   GPU·NPU 관련 업데이트

등 실무 중심의 정보가 많다.

***

## 3.4 Wired – Artificial Intelligence (요약)

Wired AI 섹션은

*   AI 기술이 사회에 미치는 영향
*   윤리·정책·규제
*   생성형 AI와 인간의 상호작용

같은 깊이 있는 분석을 제공한다. 기술뿐 아니라 산업/사회 파급력 이해에 중요하다.

***

# 4. 온디바이스 LLM 핵심 기술 요약

온디바이스 LLM은 다음 5가지 기술적 기반 위에서 가능해진다.

***

## 4.1 양자화(Quantization)

양자화는 모델 파라미터를 **FP16 → INT8 → INT4** 등으로 줄여 성능을 유지하면서

*   모델 크기 축소
*   메모리 절감
*   추론 속도 향상  
    을 만들어낸다.

대표 기법:

*   GPTQ
*   AWQ
*   SmoothQuant
*   Apple MLX Quantization
*   Qualcomm AI Engine Quantizer

***

## 4.2 프루닝(Pruning)

모델의 중요도가 낮은 연결을 제거하여 크기와 연산량을 줄이는 기법.

*   Structured pruning
*   Unstructured pruning  
    둘 다 경량화 모델에서 사용.

***

## 4.3 지식 증류(Knowledge Distillation)

대형 모델(Teacher)의 능력을 소형 모델(Student)로 전달하는 기술.  
Microsoft Phi, Samsung Gauss Nano, Google Gemini Nano 등이 이 방법을 활용한다.

***

## 4.4 경량 아키텍처(Structural Efficiency)

최근 모델은 처음부터 경량화 구조로 설계된다.

*   Phi / Phi‑3 Mini
*   Llama 3.2 1B·3B
*   Gemini Nano
*   Gauss Nano
*   Qwen2.5 Mini
*   Mamba/SSM 구조

***

## 4.5 최적화된 로컬 추론 엔진

기기별로 다른 엔진을 사용한다.

| 환경         | 추론 엔진                             |
| ---------- | --------------------------------- |
| Android    | Snapdragon AI Hub / TFLite / SNPE |
| iOS macOS  | CoreML / MLX                      |
| NVIDIA GPU | TensorRT‑LLM                      |
| CPU/PC     | GGUF + llama.cpp                  |
| Web        | WebGPU                            |

***

# 5. 글로벌 기업의 온디바이스 전략

***

## 5.1 Google

**Gemini Nano** 제공

*   멀티모달 경량 LLM
*   Android 온디바이스 API에 통합
*   TFLite 기반 최적화

## 5.2 Microsoft

**Phi 시리즈**

*   2.7B 고품질 증류 모델
*   ONNX 실행 최적화
*   PC/모바일 모두 적용 가능

## 5.3 Meta

**Llama 3.2 Small Models (1B/3B)**

*   범용 경량 LLM
*   GGUF/ONNX 모두 제공
*   로컬 실행 친화적

## 5.4 Samsung

**Gauss Nano**

*   Galaxy S24에 온디바이스 탑재
*   메시지 요약/번역/사진 기능 제공

## 5.5 Apple

**MM1 기반 온디바이스 모델 준비 중**

*   MLX, CoreML 기반
*   Apple Silicon에서 고속 실행

***

# 6. 온디바이스 LLM 모델 추천 목록 (2026 최신)

| 모델                  | 파라미터      | 특징         | 실행 포맷      |
| ------------------- | --------- | ---------- | ---------- |
| Gemini Nano 1.6B/3B | 1.6B / 3B | 멀티모달 경량 모델 | TFLite     |
| Phi‑2 / Phi‑3 Mini  | 2.7B      | 증류 기반 고성능  | ONNX       |
| Llama 3.2 1B/3B     | 1–3B      | 범용         | GGUF, ONNX |
| Qwen2.5 Mini        | 1.8B      | 한국어 강함     | GGUF       |
| Mamba/SSM 계열        | <1B       | 초경량        | MLX        |

***

# 7. 온디바이스 LLM 최적화 가이드

### ✔ 포맷 선택

*   모바일: ONNX, TFLite, CoreML
*   PC: GGUF
*   Apple Silicon: MLX

### ✔ 속도 개선

*   양자화 (INT4/INT8)
*   KV 캐시 최적화
*   슬라이딩 윈도우 어텐션

### ✔ 메모리 제한 대응

*   1B\~4B 모델 사용
*   지식 증류 적용
*   로컬 RAG 결합으로 지식 확장

***

# 8. 온디바이스 LLM FAQ

### Q1. 온디바이스에서 7B 모델 사용할 수 있나요?

→ 고성능 PC·iPhone Pro에서는 가능하지만 스마트폰에서는 대부분 3B가 현실적.

### Q2. 로컬 RAG는 어떻게 연결하나요?

→ 작은 LLM + 로컬 벡터 DB(Chroma/FAISS) 조합을 사용.

### Q3. 배터리 소모가 큰데 해결법은?

→ NPU 사용 + 양자화 + 스케줄링 최적화.

***

# 9. Copilot 에이전트에 적합한 콘텐츠 정리

이 문서(super knowledge page)는 다음 기능을 제공하기 위해 설계됨:

*   **온디바이스 LLM 핵심 개념**
*   **경량화 기술 전반**
*   **최신 글로벌 모델 동향**
*   **실제 모델 추천 리스트**
*   **최적화/실행 가이드**
*   **뉴스 요약(DeepMind · TechCrunch · VentureBeat · Wired)**
*   **FAQ**

Copilot은 이 문서를 기반으로  
온디바이스·경량화·모바일 AI 관련 질문에 **전문가 수준의 대답**을 할 수 있게 된다.

***
