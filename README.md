# HYU-Graduation-Project-Quantization
한양대학교 컴퓨터소프트웨어학부 졸업 프로젝트 진행용 레포지토리입니다.

## 자연어 처리 언어모델의 양자화와 어댑터를 활용한 memory-efficient finetuning 시스템 개발
2022-2 ~ 2023-1 한양대학교 컴퓨터소프트웨어학부 졸업 프로젝트
#### 지도교수: 서지원 <br> 팀원: 이성진, 조한빛, 황태경

### NEW! 2023-04-25 발표 자료 업로드
https://github.com/GirinMan/HYU-Graduation-Project-Quantization/blob/main/Memory_efficient_finetuning_using_quantization.pdf

### 프로젝트 개요
- 현재 자연어 처리 모델의 크기는 계속 증가하는 추세이다. 모델의 크기가 증가함에 따라 실제 서비스에서 다양한 최적화 기술들의 필요성이 대두되고 있다. 대표적인 모델 최적화 기술들 중 하나로 quantization이 있으며, 이는 실제 서비스에서의 임베디드 및 모바일 배포를 위해 자주 사용된다.
- 최근 들어 GPT-3를 비롯한 Billion scale의 모델의 사용 빈도가 매우 크게 늘었는데, 기존의 transformer-decoder 아키텍처를 그대로 사용하며 파라미터 규모와 학습 데이터를 크게 늘린 것 만으로도 다양한 NLU task에 대해 SOTA를 달성하고 few-shot 등 in-context learning이 가능해지는 등 활용 가능성이 크게 늘었다.
- 자연어 처리 모델 사이즈가 기하급수적으로 커졌기 때문에 추론 및 학습을 전용 장비가 없는 개인이 진행하기는 매우 어려워졌다. FP32 기준 10B 모델의 경우 파라미터 용량이 40GB에 육박해 일반적인 사용자용 GPU로는 학습은 커녕 추론조차 매우 어렵다.
- 이 프로젝트에서는 기존 방식으로는 사용자용 저사양 GPU에서 수행하기 어려웠던 자연어 처리 모델의 finetuning 과정을 8bit-quantization과 adapter를 활용하여 정확도 저하가 거의 없이 고성능 GPU 인프라에서 진행한 것과 거의 같은 결과를 얻을 수 있도록 하는 시스템을 개발한다. 

### 주요 목표
- LoRA adapter와 template based tuning을 활용하여 GPT 모델을 모든 task에 대해 memory-efficient하게 finetuning할 수 있도록 하는 pipeline 구축
- 같은 GPU 장치에서 모델을 8bit quantization을 적용하고 튜닝했을 때와 적용하지 않았을 때의 성능 비교
- 같은 모델, 같은 task에 대해 Quantization을 적용하여 저사양 GPU에서 학습한 결과와 고사양 GPU에서 fp16으로 학습한 결과 task 성능 비교
- 대형 언어모델의 정확도를 떨어트리지 않으면서 8bit quantization을 수행할 수 있는 LLM.int8()과 타 quantization 방식의 성능 비교

## 과제 진행 일정
- **2022 여름 방학**

  팀 구성, 주제 선정, 지도 교수 컨택
  수강신청
- **2022 2학기**

  제안서 제출(9월)
  각 지도 교수 지도 아래 졸업프로젝트 진행(1차 평가 전까지)
- **2022 겨울 방학**

  수강신청
- **2023 1학기**

  졸업프로젝트 1차평가(지도 교수 개별 평가)(4월말)

  졸업프로젝트 2차평가 (1차평가에서 하위작을 받았을 경우)(5월)

  상위작 발표회(2학기 시작 해당사항 없음)
  
  결과보고서  제출(1차평가 통과자 5월,  2차평가 통과자 6월)
