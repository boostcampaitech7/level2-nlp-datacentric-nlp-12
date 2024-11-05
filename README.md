# 주제 분류 프로젝트
> 모델 구조의 변경 없이 Data-Centric 관점으로 텍스트의 주제를 분류하는 태스크입니다.

## 소개

우리는 살아가면서 다양한 자연어 문장들을 마주하게 됩니다. 초등학교 때 쓰던 알림장에서부터, 시험 공부를 위해 들여다본 책이나, 성인이 된 후에도 계속해서 정보를 얻기 위한 글이나, 영상의 자막 모두 자연어 문장들로 이루어져 있습니다. 하다 못해 지인들과 주고 받는 메세지와 편지들, 업무 전달을 위한 메신저와 문서들도 모두 자연어 문장들로 이루어져 있습니다. 어렸을 때부터 우리는 무의식적으로 각 자연어 문장들이 어떤 주제로 이루어져 있는지 판단 후 내용을 파악하게 됩니다.

![주제 분류 시각화 (출처: https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a)](https://aistages-api-public-prod.s3.amazonaws.com/app/Users/00002865/files/1b553a15-90c8-4791-b20f-6f25f1d6e112..png)

그렇다면 사람이 아니라 딥러닝 모델은 어떨까요?

자연어를 독해 및 분석 과정을 거쳐 주어진 태스크를 수행하기 위해서는 자연어의 주제에 대한 이해가 필수적입니다. Topic Classification 태스크는 모델이 자연어를 잘 이해하고 있는지 평가할 수 있는 가장 간단한 태스크입니다.

그 중에서도 KLUE-Topic Classification benchmark는 뉴스의 헤드라인을 통해 그 뉴스가 어떤 topic을 갖는지를 분류해내는 태스크입니다. 각 자연어 data에는 생활문화(Society), 스포츠(Sports), 세계(World), 정치(Politics), 경제(Economy), IT과학(IT/Science), 사회(Society) 등 다양한 주제 중 하나가 라벨링 되어 있습니다.

본 대회는 결과물 csv 확장자 파일을 제출하게 됩니다.

- input : 약 30,000개의 뉴스 헤드라인과 url, 작성 날짜
- output : 각 뉴스 헤드라인의 주제 (생활문화, 스포츠, 세계, 정치, 경제, IT과학, 사회 중 하나. 정수로 인코딩된 값)

Data-Centric 의 취지에 맞게, 베이스라인 모델의 수정 없이 오로지 data의 수정으로만 성능 향상을 이끌어내야 합니다. 베이스라인 코드의 수정이 없는 선에서, 또한 유료 버전의 API 사용 없는 선에서 모든 방법을 적용할 수 있습니다. 대회 종료 후 베이스라인 코드의 변경이 확인되는 경우, 리더보드에서 제외됩니다.

## 가능한 방법

- 공개된 생성 모델 (T5 등)을 통한 Synthetic Data 생성
- 각종 Data Augmentation 기법 적용
- Data sampling 기법
- Negative sampling 등
- Data Quality Control
- Data labeling error detection, Data Cleaning, Data Filtering 등
- 다양한 Self-training 기법 (code가 아닌 data 변경)
- 모델의 변경이 아닌 베이스라인 코드의 변경
- 학습 소모 시간 변경을 위한 batch size 및 max sequence length
- Data 크기 변경을 위한 train&valid dataset split 비율
- batch size 조정 등에 의한 learning rate 값 조절

## 불가능한 방법

- 유료 버전의 비공개 생성 모델을 활용하는 모든 방법 (GPT-4, ChatGPT (GPT-3.5), api key 결제 일체 포함)
- 베이스라인 코드 내 모델의 변경을 야기하는 모든 방법들
- [가능한 방법]에 기재되지 않은 모든 베이스라인 코드의 변경
- 베이스라인 모델 변경 불가
- 단, 제한된 서버 자원 내에서 공개된 외부 모델을 사용하여 data 수정은 가능
- 테스트셋 정보를 활용한 모든 방법
- 외부 dataset 사용
- Active learning, Curriculum learning 등의 data 수정이 아닌 모델을 업데이트 하는 방법
- Test dataset에 대한 전처리

# 데이터 개요

대회에서 사용되는 데이터셋은 KLUE 공식 사이트에서 제공하는 KLUE-TC(YNAT) 데이터셋과 같은 포맷을 가집니다. 제공되는 총 학습 데이터는 2,800개이며, 테스트 데이터는 30,000개 입니다.

기존 KLUE-YNAT 데이터셋 중 일부를 학습데이터로, 일부를 테스트데이터로 사용합니다.

- 대부분의 학습데이터에는 noise (text 컬럼 값에 노이즈 및 라벨 변경)가 섞인 데이터가 일부 섞여있습니다.
- 노이즈가 심한 데이터를 필터링하고, 노이즈가 심하지 않은 데이터를 복구하고, 잘못된 라벨이 있는 데이터를 찾아서 고치거나 필터링하는 것이 대회의 핵심입니다.
- 데이터 증강을 진행할 때, 노이즈가 심한 데이터 혹은 잘못된 라벨이 있는 데이터를 유의해서 진행해야 합니다.

데이터는 아래와 같이 csv 파일로 제공되며, 각 행이 하나의 데이터 샘플입니다. 최대한 깔끔하고 명확한 대회 진행을 위해, KLUE-YNAT 데이터 중 일부 feature만을 사용합니다.

![KLUE-YNAT 데이터 발췌](https://aistages-api-public-prod.s3.amazonaws.com/app/Files/74fafe34-ea5f-4d7f-92de-4b64dde2892a.png)

- `ID`: 각 데이터 샘플의 고유번호 입니다.
- `text`: 분류의 대상이 되는 자연어 텍스트입니다. 연합 뉴스 기사의 헤드라인이며, 한국어 텍스트에 일부 영어, 한자 등의 단어가 포함되어 있습니다.
- `target`: 정수로 인코딩 된 라벨입니다.

# 코드 개요

주어진 code.tar.gz 파일의 압축을 해제하면 다음과 같은 구조로 파일이 존재합니다

```bash
├─code
│  │  baseline.ipynb
│  │  requirements.txt
```

- `baseline.ipynb` : 베이스라인 코드입니다. 크게 7가지로 분류됩니다.

1. Load Libraries: 베이스라인 코드 실행에 필요한 모든 라이브러리를 로드합니다.

2. Set Hyperparameters: CUDA 사용을 위한 device 설정과 경로 지정, 시드 설정을 수행합니다. 또한 max sequence length, batch size 등의 hyperparameter도 지정합니다.

3. Load Tokenizer and Model: 사전 학습된 tokenizer와 model을 로드합니다. 본 베이스라인 코드에서는 klue/bert-base (https://huggingface.co/klue/bert-base)를 사용합니다.

4. Define Dataset: train.csv 파일을 로드하고, train dataset과 dev dataset을 7:3의 비율로 split합니다. BERTDataset class를 정의하고, tokenizing과 padding을 적용하기 위한 data_collector 를 선언합니다.

5. Define Metric: 대회의 metric인 f1_score를 정의하기 위한 함수를 선언합니다.

6. Train Model Define Model : hyperparameter, metric, data loader를 반영하여 학습할 수 있도록 TrainingArgument 모듈로 training arguments를 정의하고, Trainer 모듈을 사용하여 학습을 진행합니다.

7. Evaluate Model: test.csv 파일을 업로드한 후 data loader에 할당해줍니다. model을 evaluation mode로 변경한 후 model의 입력으로 제공하여 예측값을 계산합니다.

- `requirements.txt` : 코드 실행을 위해 필수적으로 설치되어야 하는 패키지의 이름과 그 버전이 적혀있습니다. 본 베이스라인 코드에서는 transformers, sentencepiece, numpy, pandas, evaluate, accelerate, scikit-learn, ipywidgets, protobuf, torch가 필요합니다. 단, torch는 아래의 별도의 명령어로 설치해야 합니다.

실행을 위해서는 먼저 `pip install -r requirements.txt`로 패키지를 먼저 설치한 후, baseline.ipynb의 셀을 순서대로 실행하면 됩니다. 이 때, `BASE_DIR`의 경로를 현재 사용자의 경로로 지정해주시면 됩니다.

```bash
$ pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
$ pip install -r requirements.txt
```

코드를 모두 실행하고 나오는 결과물은 `BASE_DIR`내의 output.csv 의 파일로 저장됩니다. 이 파일은 기존 test.csv 파일에, 예측 결과가 함께 저장되어 있습니다.

베이스라인 성능은 아래와 같습니다

- 'accuracy': 0.7111, 'f1': '0.6975' (train, valid split 비율 0.2의 경우)
