
# 📘 Qwen3‑4B‑Instruct‑2507 — One‑Page 모델 노트

## 1. 모델 개요

Qwen3‑4B‑Instruct‑2507은 **Qwen3‑4B Non‑Thinking 모델의 업데이트 버전**으로,  
일반 지능·추론·멀티링구얼·정렬(Alignment)·에이전트 능력을 대폭 개선한 **4B급 인스트럭트 모델**.

### 주요 특징

*   ✔ **Non‑Thinking 전용 모델** (즉, `<think>` 블록 없음 / `enable_thinking=False` 불필요)
*   ✔ 지시 따르기, 세계 지식, 창의적 글쓰기, 수학·과학·코딩 성능 크게 향상
*   ✔ 256K **초장문 문맥(Long‑context) 네이티브 지원**
*   ✔ 다국어 범위 확장 및 long‑tail knowledge 강화
*   ✔ 도구 호출(agentic tool use) 강화 → Qwen-Agent와 연동 최적화

***

## 2. 모델 구조 & config 기반 요약

### 📊 핵심 스펙

| 항목              | 값                       |
| --------------- | ----------------------- |
| 파라미터 수          | 4.0B                    |
| (임베딩 제외)        | 3.6B                    |
| 레이어 수           | 36                      |
| Attention (GQA) | 32 Q‑heads / 8 KV‑heads |
| Hidden size     | 2560                    |
| FFN size        | 9728                    |
| Context length  | **262,144 tokens**      |
| Activation      | SiLU                    |
| Norm            | RMSNorm (eps 1e‑6)      |
| RoPE θ          | 5,000,000               |
| Vocab           | 151,936                 |
| Architecture    | `Qwen3ForCausalLM`      |

### 아키텍처 특징

*   **Full attention** 기반
*   RoPE scaling 비활성(기본 RoPE)
*   대규모 θ(5M)를 사용하여 초장문 성능 최적화
*   Sliding‑window 없음 (use\_sliding\_window=False)

***

## 3. 성능 요약(핵심만)

### ✨ Qwen3‑4B → Qwen3‑4B‑Instruct‑2507 업그레이드 폭발적 향상

| Benchmark           | 기존 4B | 4B‑Instruct‑2507 |
| ------------------- | ----: | ---------------: |
| MMLU‑Pro            |  58.0 |         **69.6** |
| MMLU‑Redux          |  77.3 |         **84.2** |
| GPQA                |  41.7 |         **62.0** |
| AIME‑25             |  19.1 |         **47.4** |
| HMMT‑25             |  12.1 |         **31.0** |
| ZebraLogic          |  35.2 |         **80.2** |
| IFEval              |  81.2 |         **83.4** |
| Creative Writing v3 |  53.6 |         **83.5** |
| BFCL‑v3             |  57.6 |         **61.9** |
| TAU‑Retail          |  24.3 |         **48.7** |

→ **지시 수행, 추론, 창의 글쓰기, 에이전트까지 모든 축에서 4B급 최고 수준**

***

## 4. 기본 사용법(Quickstart)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer([text], return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=16384)

print(tokenizer.decode(output[0][len(inputs.input_ids[0]):], skip_special_tokens=True))
```

※ `transformers >= 4.51.0` 필수  
(`KeyError: 'qwen3'` 방지)

***

## 5. 에이전트/툴 사용(Qwen-Agent 권장)

```python
from qwen_agent.agents import Assistant

llm_cfg = {
    'model': 'Qwen3-4B-Instruct-2507',
    'model_server': 'http://localhost:8000/v1',
    'api_key': 'EMPTY',
}

tools = [
    {'mcpServers': {
        'time': {'command': 'uvx', 'args': ['mcp-server-time', '--local-timezone=Asia/Shanghai']},
        'fetch': {'command': 'uvx', 'args': ['mcp-server-fetch']}
    }},
    'code_interpreter',
]

bot = Assistant(llm=llm_cfg, function_list=tools)

messages = [{'role': 'user', 'content': 'Introduce the latest developments of Qwen'}]
for out in bot.run(messages=messages):
    pass
print(out)
```

Qwen-Agent는:

*   Tool‑calling 템플릿/파서 자동 포함
*   MCP 기반 tool 구성 지원  
    → **코드 복잡도 크게 감소**

***

## 6. 배포(Deployment)

### SGLang (0.4.6.post1 이상)

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B-Instruct-2507 \
  --context-length 262144
```

### vLLM (0.8.5 이상)

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 262144
```

➡ OOM 발생 시 context length를 32K 등으로 축소

***

## 7. Best Practices (권장 설정)

### Sampling

*   `temperature = 0.7`
*   `top_p = 0.8`
*   `top_k = 20`
*   필요 시 `presence_penalty ∈ [0, 2]`  
    (높을수록 반복 억제, 그러나 언어 혼합 가능)

### Output length

*   권장: **16,384 tokens**

### Benchmarking 시 권장 프롬프트

*   **수학** → “Please reason step by step, and put your final answer within \boxed{}.”
*   **객관식** → JSON 구조 포함 `"answer": "C"`

***

## 8. Citation

    @misc{qwen3technicalreport,
          title={Qwen3 Technical Report},
          author={Qwen Team},
          year={2025},
          eprint={2505.09388},
          archivePrefix={arXiv},
          primaryClass={cs.CL},
          url={https://arxiv.org/abs/2505.09388},
    }

***

# 📘 EXAONE‑4.0‑1.2B — One‑Page 모델 노트

## 1. 모델 개요

**EXAONE 4.0** 시리즈는

*   **Non‑reasoning 모드**(일반 대화/지시 수행)와
*   **Reasoning 모드**(논리·수학·추론)  
    두 기능을 **하나의 모델에 통합한 최초의 EXAONE 라인업**.

**1.2B 모델**은 **온디바이스용 경량 모델**로 개발되었으며  
한국어·영어·스페인어까지 자연스럽게 지원.

### 주요 특징

*   Reasoning과 Non‑reasoning 통합
*   에이전트 도구 호출(Agentic tool use) 지원
*   한국어 실용 지식(고난도 포함) 성능 강화
*   긴 문맥 처리: **65,536 tokens**
*   소형 모델 대비 상위권 세계 지식/수학/도구 호출 성능

***

## 2. 아키텍처 & 구성(config.json 기반)

### 📊 핵심 스펙

| 항목                  | 값                                      |
| ------------------- | -------------------------------------- |
| 파라미터 수              | 1.07B (임베딩 제외)                         |
| 레이어                 | 30                                     |
| Attention           | GQA (32 heads / 8 KV-heads)            |
| Hidden size         | 2048                                   |
| Intermediate size   | 4096                                   |
| Vocab size          | 102,400                                |
| Context length      | **65,536**                             |
| Positional Encoding | RoPE (Llama3 style, scaling factor 16) |
| Activation          | SiLU                                   |
| Normalization       | RMSNorm + QK-Reorder-Norm              |
| Dtype               | bfloat16                               |
| Architecture        | `Exaone4ForCausalLM`                   |

### 🔧 구조적 변화(4.0의 핵심 차별점)

*   **QK‑Reorder‑Norm**: Q/K projection 직후 RMSNorm 적용 → 추론·지시 성능 향상
*   **Full Attention(1.2B)**: 소형 모델 특성상 hybrid 대신 Full attention 사용
*   RoPE scaling으로 초장문 처리 능력 확보

***

## 3. 사용 가이드 (Quickstart)

### 3.1 일반 모드(Non‑reasoning)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="bfloat16",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [{"role": "user", "content": "Explain how wonderful you are"}]
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
)

output = model.generate(
    input_ids.to(model.device),
    max_new_tokens=128,
    do_sample=False
)
print(tokenizer.decode(output[0]))
```

***

### 3.2 Reasoning 모드

`enable_thinking=True` → `<think>` 블록을 열고 Reasoning 활성화

```python
messages = [{"role": "user", "content": "Which one is bigger, 3.12 vs 3.9?"}]
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt", enable_thinking=True
)

output = model.generate(
    input_ids.to(model.device),
    max_new_tokens=128,
    do_sample=True, temperature=0.6, top_p=0.95
)
print(tokenizer.decode(output[0]))
```

※ Reasoning 모드는 **sampling 파라미터 영향이 매우 큼**

***

### 3.3 Agentic Tool Use (도구 호출)

```python
def roll_dice(max_num: int):
    return random.randint(1, max_num)

tools = [{
    "type": "function",
    "function": {
        "name": "roll_dice",
        "description": "Roll a dice",
        "parameters": {
            "type": "object",
            "required": ["max_num"],
            "properties": {"max_num": {"type": "int"}}
        }
    }
}]

messages = [{"role": "user", "content": "Roll D6 dice twice!"}]
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt", tools=tools
)

output = model.generate(
    input_ids.to(model.device),
    max_new_tokens=1024,
    do_sample=True, temperature=0.6, top_p=0.95
)
print(tokenizer.decode(output[0]))
```

***

## 4. 배포(Deployment)

### TensorRT‑LLM

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
```

추가 설정 파일 예시:

```yaml
# extra_llm_api_config.yaml
kv_cache_config:
  enable_block_reuse: false
```

서버 실행:

```bash
trtllm-serve serve LGAI-EXAONE/EXAONE-4.0-1.2B \
  --backend pytorch \
  --extra_llm_api_options extra_llm_api_config.yaml
```

***

### vLLM (0.10.0 이상)

```bash
vllm serve LGAI-EXAONE/EXAONE-4.0-1.2B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser deepseek_r1
```

***

## 5. 성능 요약(1.2B 기준 핵심만)

### Reasoning Mode (1.2B)

*   **MMLU‑Redux**: 71.5
*   **AIME‑2025**: 45.2 (소형 모델 중 우수)
*   **BFCL‑v3(Tool use)**: 52.9
*   **KMMLU‑Redux(한국어)**: 46.9
*   **MMMLU(ES)**: 62.4

### Non‑Reasoning Mode (1.2B)

*   **MMLU‑Redux**: 66.9
*   **IFEval**: 74.7 (지시 수행 매우 강함)
*   **Long context (RULER)**: 77.4
*   **Ko‑LongBench**: 69.8

👉 **경량 모델 중 가장 균형 잡힌 성능(지식·추론·한국어·스페인어·도구 사용)**

***

## 6. 추천 사용 설정 (Usage Guideline)

| 모드              | 권장 설정                                 |
| --------------- | ------------------------------------- |
| Non‑reasoning   | `temperature < 0.6`                   |
| Reasoning       | `temperature=0.6`, `top_p=0.95`       |
| Degeneration 방지 | `presence_penalty=1.5`                |
| 한국어 일반 대화(1.2B) | `temperature=0.1` (code-switching 방지) |

***

## 7. 라이선스 요약

**EXAONE AI Model License Agreement 1.2 – NC**

*   출력물 소유권 제한 조항 삭제됨
*   **경쟁 모델 개발용 사용 금지**
*   연구 + 교육 목적 허용

***

## 8. 참고

Citation:

    @article{exaone-4.0,
      title={EXAONE 4.0: Unified Large Language Models Integrating Non-reasoning and Reasoning Modes},
      author={{LG AI Research}},
      journal={arXiv preprint arXiv:2507.11407},
      year={2025}
    }

***

# 📘 Qwen3‑1.7B — One‑Page 모델 노트

## 1. 모델 개요

**Qwen3‑1.7B**는 Qwen3 세대의 **경량 중형 모델**로,  
다음 특징이 핵심입니다:

### 주요 기능 요약

*   ✔ **Thinking ↔ Non‑Thinking 모드 완전 통합**  
    → 하나의 모델에서 논리 추론/수학/코딩 강화 모드와 빠른 일반 대화 모드를 전환 가능
*   ✔ QwQ·Qwen2.5 대비 **대폭 향상된 추론 능력**
*   ✔ 풍부한 인간 선호 정렬(Alignment)  
    → 창작, 롤플레잉, 멀티턴 대화 등 자연스러운 상호작용
*   ✔ **Agent(도구 호출) 강화**  
    → reasoning/비‑reasoning 모두에서 도구 호출 동작
*   ✔ 100+ 언어 지원 (번역 및 다국어 인스트럭션 성능 강화)

***

## 2. 모델 구조 (config 기반)

### 📊 핵심 스펙

| 항목                | 값                       |
| ----------------- | ----------------------- |
| 파라미터              | 1.7B                    |
| (임베딩 제외)          | 1.4B                    |
| 레이어               | 28                      |
| Attention (GQA)   | 16 Q-heads / 8 KV-heads |
| Hidden size       | 2048                    |
| FFN(intermediate) | 6144                    |
| Context length    | **32,768**              |
| RoPE θ            | 1,000,000               |
| Activation        | SiLU                    |
| Norm              | RMSNorm (eps=1e‑6)      |
| Architecture      | `Qwen3ForCausalLM`      |
| Vocab             | 151,936                 |
| Dtype             | bfloat16                |
| Sliding-window    | 없음                      |

***

## 3. Thinking/Non‑Thinking 모드

### enable\_thinking=True (기본)

*   `<think> ... </think>` 블록 생성
*   math/logic/code에서 최적 성능
*   권장 Sampling:
    *   `temperature=0.6`, `top_p=0.95`, `top_k=20`

### enable\_thinking=False

*   `<think>` 블록 완전 비활성
*   빠른 inference/일반 인스트럭션 최적
*   권장 Sampling:
    *   `temperature=0.7`, `top_p=0.8`, `top_k=20`

### Soft switch (유저 입력 기반)

*   `/think` → 해당 턴부터 Thinking
*   `/no_think` → 해당 턴 Non‑Thinking
*   enable\_thinking=True일 때만 동작
*   enable\_thinking=False면 soft switch 무시됨

***

## 4. Quickstart (Thinking 모드 예시)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # 기본값
)

inputs = tokenizer([text], return_tensors="pt").to(model.device)
gen = model.generate(**inputs, max_new_tokens=32768)[0]

# thinking/answer 분리
out = gen[len(inputs.input_ids[0]):].tolist()
try:
    idx = len(out) - out[::-1].index(151668)  # </think>
except ValueError:
    idx = 0

thinking = tokenizer.decode(out[:idx], skip_special_tokens=True).strip()
answer   = tokenizer.decode(out[idx:], skip_special_tokens=True).strip()

print("thinking:", thinking)
print("answer:", answer)
```

***

## 5. Thinking ↔ Non‑Thinking 전환 예시

### Non‑Thinking 모드

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
```

### Soft switching 예시

    User: How many r's in strawberries?
    User: Then how many r's in blueberries? /no_think
    User: Really? /think

***

## 6. Agentic Tool Use (Qwen-Agent)

```python
from qwen_agent.agents import Assistant

llm_cfg = {
    'model': 'Qwen3-1.7B',
    'model_server': 'http://localhost:8000/v1',
    'api_key': 'EMPTY'
}

tools = [
    {'mcpServers': {
        'time': {'command': 'uvx', 'args': ['mcp-server-time', '--local-timezone=Asia/Shanghai']},
        'fetch': {'command': 'uvx', 'args': ['mcp-server-fetch']}
    }},
    'code_interpreter'
]

bot = Assistant(llm=llm_cfg, function_list=tools)

for res in bot.run(messages=[{"role":"user","content":"Introduce Qwen's latest progress"}]):
    pass
print(res)
```

***

## 7. 배포(Deployment)

### SGLang

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-1.7B \
  --reasoning-parser qwen3
```

### vLLM

```bash
vllm serve Qwen/Qwen3-1.7B \
  --enable-reasoning \
  --reasoning-parser deepseek_r1
```

***

## 8. Best Practices

### Thinking 모드

*   `temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0`
*   ❌ Greedy decoding 금지 (성능 저하·반복 발생)

### Non‑Thinking 모드

*   `temperature=0.7`, `top_p=0.8`, `top_k=20`

### 반복 억제

*   `presence_penalty ∈ [0, 2]`  
    (높이면 언어 혼합 가능)

### 출력 길이

*   권장: **32,768 tokens**
*   고난도 math/coding 벤치마크: **38,912 tokens**

### 멀티턴 시 Best practice

*   히스토리에는 **최종 답변만** 저장
*   `<think>` 내용은 히스토리에 포함 X

***

## 9. Citation

    @misc{qwen3technicalreport,
          title={Qwen3 Technical Report},
          author={Qwen Team},
          year={2025},
          eprint={2505.09388},
          archivePrefix={arXiv},
          primaryClass={cs.CL},
          url={https://arxiv.org/abs/2505.09388},
    }

***

# 📘 EXAONE‑3.5‑2.4B‑Instruct — One‑Page 모델 노트

## 1. 모델 개요

\*\*EXAONE 3.5(2.4B)\*\*는 LG AI Research가 공개한 **영·한 이중언어(English/Korean) 인스트럭트 LLM**으로,  
다음 특성을 갖는 **소형·상용 배포 최적화 모델**입니다.

### 주요 특징

*   ✔ **2.4B 경량 모델** — 작은 GPU/온디바이스 방향
*   ✔ **32K 토큰** 장문 지원
*   ✔ 영어+한국어 자연스러운 이중언어 모델
*   ✔ 실사용 중심 성능 최적화 (MT-Bench, LiveBench, KoMT-Bench 등 우수)
*   ✔ Word embedding tied (7.8B/32B는 untied)
*   ✔ 실제 서비스/챗봇에 바로 적용 가능한 안정적 인스트럭션 튜닝

***

## 2. 모델 구조(config.json 기반)

### 📊 핵심 스펙

| 항목                  | 값                             |
| ------------------- | ----------------------------- |
| 파라미터(Non‑Embedding) | **2.14B**                     |
| 레이어 수               | 30                            |
| Hidden size         | 2560                          |
| FFN(intermediate)   | 7168                          |
| Attention           | GQA (32 Q‑heads / 8 KV‑heads) |
| Head dim            | 80                            |
| Context length      | **32,768**                    |
| Activation          | SiLU                          |
| Positional encoding | RoPE (Llama3 style, factor=8) |
| Norm                | LayerNorm (eps 1e‑5)          |
| Vocab size          | 102,400                       |
| Embedding tied      | **True**                      |
| Architecture        | `ExaoneForCausalLM`           |

→ **메모리 대비 성능 효율이 매우 좋도록 설계된 구조**

***

## 3. Quickstart (공식 예제 요약)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "system", "content": "You are EXAONE model from LG AI Research, a helpful assistant."},
    {"role": "user", "content": "스스로를 자랑해 봐"}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

output = model.generate(
    input_ids.to(model.device),
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=128,
    do_sample=False
)
print(tokenizer.decode(output[0]))
```

### 중요!

*   **EXAONE 3.5는 system prompt 최적화 학습됨 → 반드시 system role 사용 권장**

***

## 4. 성능 요약(대표 지표)

| 모델                  | MT‑Bench | LiveBench | Arena‑Hard | AlpacaEval |   IFEval | KoMT‑Bench | LogicKor |
| ------------------- | -------: | --------: | ---------: | ---------: | -------: | ---------: | -------: |
| **EXAONE 3.5 2.4B** | **7.81** |  **33.0** |   **48.2** |   **37.1** | **73.6** |   **7.24** | **8.51** |
| Qwen2.5 3B          |     7.21 |      25.7 |       26.4 |       17.4 |     60.8 |       5.68 |     5.21 |
| Qwen2.5 1.5B        |     5.72 |      19.2 |       10.6 |        8.4 |     40.7 |       3.87 |     3.60 |
| Llama 3.2 3B        |     6.94 |      24.0 |       14.2 |       18.7 |     70.1 |       3.16 |     2.86 |
| Gemma2 2B           |     7.20 |      20.0 |       19.1 |       29.1 |     50.5 |       4.83 |     5.29 |

→ **특히 한국어 실사용 지표(KoMT‑Bench, LogicKor)에서 동급 최고 성능**

***

## 5. 배포(Deployment)

지원되는 프레임워크:

*   **TensorRT‑LLM**
*   **vLLM**
*   **SGLang**
*   **llama.cpp**
*   **Ollama**

→ 소형 모델 특성상 **vLLM / SGLang / GGUF(quant)** 조합이 가장 실전 최적화

***

## 6. Quantization(양자화)

LG AI Research에서 **AWQ / GGUF** 양자화 모델 제공

*   2/3/4-bit 등 다양한 양자화 옵션
*   cpu/on‑device 환경에서도 실사용 가능

→ “EXAONE 3.5 collection” 페이지 참고

***

## 7. 모델 사용 팁 (Best Practices)

*   **system prompt 반드시 포함**  
    → 학습 과정에서 system role 정보를 적극 반영함
*   do\_sample=False 시 매우 안정적인 출력
*   한국어 대화에서 높은 일관성
*   32K context를 활용해 문서 요약, RAG 기반 추론 등에 적합

***

## 8. 제한 사항

*   최신 정보 반영 X → 현실 세계 최신 데이터는 틀릴 수 있음
*   학습 데이터 기반 편향 존재 가능
*   잘못된 문장 생성, 불완전한 추론 가능
*   민감한 내용에 대해 부적절 응답 가능성 → 사용자 검증 필요

***

## 9. 라이선스

**EXAONE AI Model License Agreement 1.1 — NC**

*   비상업적 사용 중심
*   세부 내용은 레포지토리 License 참고

***

## 10. Citation

    @article{exaone-3.5,
      title={EXAONE 3.5: Series of Large Language Models for Real-world Use Cases},
      author={LG AI Research},
      journal={arXiv preprint arXiv:2412.04862},
      year={2024}
    }

***
