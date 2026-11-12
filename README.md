# Bank Account Fraud Detection

은행계좌 개설 신청의 사기 여부를 예측하는 개인 심층학습 연구입니다. BAF Base에서 손실함수와 신경망 구조의 효과를 비교하며, 시간 분할과 재현 가능한 실행 기록을 사용합니다. 연구자는 황정현입니다.

## 설치

Python 3.11~3.14를 사용합니다. 로컬 Mac에서는 다음 명령으로 프로젝트 전용 환경을 구성합니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev]'
```

데이터 검증과 전처리에는 PyTorch가 필요하지 않습니다. Mac에서 장치를 확인하려면 `python -m pip install '.[torch]'`를 추가로 실행합니다. Linux GPU 서버의 PyTorch 설치는 서버의 드라이버와 GPU 사양을 확인한 뒤 정합니다. 특정 CUDA 버전은 미리 지정하지 않습니다.

`pyproject.toml`은 의존성 범위를 정의합니다. `requirements-local.lock.txt`는 검증에 사용한 Mac 환경의 정확한 패키지 버전이며, Linux CUDA 환경의 설치 목록으로 사용하지 않습니다. 같은 Mac 환경을 재구성할 때는 `python -m pip install -r requirements-local.lock.txt`와 `python -m pip install --no-deps .`를 실행합니다.

소스를 고친 뒤에는 프로젝트를 다시 설치합니다. 재설치 없이 확인하려면 저장소 루트에서 `PYTHONPATH=src python scripts/prepare_data.py --help`처럼 실행합니다.

## 데이터 다운로드

공식 배포처에서 Base 버전의 `Base.csv`를 내려받아 `data/raw/Base.csv`에 둡니다. [공식 GitHub](https://github.com/feedzai/bank-account-fraud)는 [Kaggle 다운로드 페이지](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)를 안내합니다. 다운로드와 압축 해제를 마친 뒤 Base와 Variant 파일을 혼동하지 않았는지 확인합니다.

CLI는 터미널에서 명령어로 실행하는 인터페이스입니다. 이 프로젝트의 데이터 CLI는 CSV만 지원합니다.

```bash
mkdir -p data/raw
# Kaggle 페이지에서 받은 Base.csv를 data/raw/Base.csv로 복사합니다.
baf-data --config configs/phase1.toml --audit-only \
  --report-dir results/phase1/audit
```

Kaggle CLI를 이미 설치하고 공식 인증을 설정했다면 다음 방법도 사용할 수 있습니다. 데이터셋 전체 압축 파일에는 추가 Variant가 들어갈 수 있으므로, 전처리에는 Base만 지정합니다.

```bash
kaggle datasets download \
  -d sgpjesus/bank-account-fraud-dataset-neurips-2022 -p data/raw
```

인증키는 저장소에 두지 않습니다. `.gitignore`는 `data/`, `checkpoints/`, 데이터 파일, 모델 가중치와 인증 파일을 추적에서 제외합니다. `results/`의 집계 CSV만 예외로 허용하며, 원본 행과 전처리 행은 이곳에 저장하지 않습니다.

데이터의 규모·사기 비율·월별 행 수는 다운로드한 파일에서 측정합니다. 알려진 근삿값을 검증 결과로 기록하지 않습니다.

CSV에 저장용 인덱스가 있으면 연속된 0 기반 인덱스인지 확인한 뒤 제거합니다. 인덱스가 없으면 첫 번째 정답 열을 그대로 보존합니다.

32개 열의 이름과 값의 기본 조건을 확인하지만, 같은 스키마를 가진 Variant와 Base를 내용만으로 구별하지는 못합니다. 파일의 출처와 이름도 확인해야 합니다.

## 데이터 사용 조건

배포 출처 사이에 라이선스 표기가 다릅니다. [공식 datasheet의 Q44와 Author Statement](https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf)는 CC BY-NC-ND 4.0을 명시하지만, 2026년 11월 5일 조회한 [Kaggle 공개 메타데이터](https://www.kaggle.com/api/v1/datasets/list?search=bank-account-fraud-dataset-neurips-2022)의 `licenseName`은 CC BY-NC-SA 4.0입니다.

어느 표기가 우선하는지 단정하지 않습니다. 원본·전처리 데이터는 Git에서 제외하고, 재배포나 상업 이용을 검토한다면 배포자에게 적용 조건을 확인합니다. 저장소 코드의 라이선스를 데이터의 라이선스로 대신 적용하지 않습니다.

출처를 명시하고 연구 결과에서 데이터셋 논문을 인용합니다. 합성 데이터에서의 결과를 실제 은행 서비스의 성능으로 일반화하지 않습니다. 공식 설명서도 실제 은행 사기 탐지에 직접 사용하는 것을 피하도록 안내합니다.

Jesus et al. (2022), [Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation](https://arxiv.org/abs/2211.13358), NeurIPS Datasets and Benchmarks.

## 검증과 전처리

실행 설정은 `configs/phase1.toml`에서 관리합니다. 설정 파일 안의 상대 경로는 `configs/`의 상위 디렉터리를 기준으로 해석합니다. 명령행으로 덮어쓴 상대 경로는 실행한 디렉터리를 기준으로 해석합니다.

```bash
# 실제 Base 전체를 검증하고 전처리합니다.
baf-data --config configs/phase1.toml

# 모든 월을 포함하는 작은 표본으로 전처리 동작을 확인합니다.
baf-data --config configs/phase1.toml --sample-per-month 1000 \
  --processed-dir data/processed/smoke \
  --report-dir results/phase1/smoke

# 데이터가 다른 곳에 있으면 경로를 덮어씁니다.
baf-data --config configs/phase1.toml --data /path/to/Base.csv \
  --processed-dir data/processed/alternate \
  --report-dir results/phase1/alternate
```

데이터가 없으면 다운로드 주소를 포함한 오류와 종료 코드 2를 반환합니다. 기존 출력 폴더가 있으면 덮어쓰지 않고 중단합니다.

같은 입력·시드·설정은 같은 표본과 변환값을 만듭니다. 재실행 결과를 따로 보관하려면 새 출력 경로를 지정합니다. 실행 시각과 소요 시간은 실행마다 달라집니다.

작은 표본 모드도 전체 CSV의 스키마와 통계를 먼저 검증합니다. 각 월에서 고정 시드로 균일 표본을 추출하며, 이 표본은 전체 실험 결과로 사용하지 않습니다. 라벨 비율을 인위적으로 맞추지 않으므로 작은 표본에는 특정 정답 클래스가 없을 수 있습니다. `manifest.json`은 전체 파일의 집계와 실제 전처리한 표본의 행 수를 구분합니다.

| 분할 | 월 | 용도 |
|---|---|---|
| Train | 0~5 | 전처리 기준과 모델 가중치를 학습합니다. |
| Validation | 6 | 모델·설정·판정 임계값을 선택합니다. |
| Test | 7 | 선정한 모델과 고정 임계값으로 최종 평가합니다. |

`fraud_bool`과 `month`는 모델 입력에서 제외합니다. 범주형·이진·수치형 구분은 [공식 datasheet Q7](https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf)에 근거하며, 실제 파일의 데이터 타입과 값도 검사합니다. 누락되거나 추가된 열, 잘못된 정답·월·이진 값, 무한대와 비수치형 값은 오류로 처리합니다. 전체 파일에 월 0~7이 모두 있어야 하며, 향후 평가를 위해 각 분할에 두 정답 클래스가 있어야 합니다.

특수 결측값은 열별로 변환합니다. `prev_address_months_count`, `current_address_months_count`, `bank_months_count`, `session_length_in_minutes`, `device_distinct_emails`는 `-1`을 NaN으로 바꿉니다. `intended_balcon_amount`는 모든 음수를 NaN으로 바꿉니다. 다른 열의 음수는 일괄 변환하지 않습니다.

공식 `Base.csv`의 `device_distinct_emails_8w`는 CSV를 읽을 때 Datasheet의 `device_distinct_emails`로 연결합니다. 실행 기록에 원본 헤더와 적용한 열 이름 대응을 남깁니다.

수치형은 Train 중앙값으로 채우고 Train에서 학습한 `StandardScaler`로 표준화합니다. 이진 특성은 Train 최빈값으로 채웁니다. Train에서 전부 결측인 수치형·이진 열은 0으로 채웁니다.

Datasheet에서 특수 결측값을 명시한 수치형 여섯 열에는 항상 결측 표시 열을 만듭니다. 나머지 수치형과 이진형은 Train에서 결측이 관측된 열에 표시를 추가합니다. 결측 표시 열, 이진 특성, 원핫 인코딩 결과는 0/1을 유지합니다. Validation이나 Test에서 Train에 없던 결측이 나타나면 분할·열 이름·결측 건수를 경고 로그에 남기고 기존 입력 차원으로 변환합니다.

범주형은 고정 문자열 `__MISSING__`로 채운 뒤 Train에서 학습한 원핫 인코딩을 적용합니다. 원핫 인코딩은 범주마다 별도 열을 만드는 방식입니다. 미래에 처음 등장한 범주는 해당 특성의 인코딩 열을 모두 0으로 표시합니다.

XAI는 모델의 예측 근거를 분석하는 설명 가능한 AI 방법을 뜻합니다. 순서가 없는 범주에 숫자 순서가 생기지 않고, XAI에서 원래 특성으로 묶을 수 있어 이 방식을 선택했습니다.

메모리 보호를 위해 범주형 열의 고유값이 100개를 넘으면 중단합니다. 이 한도는 구현의 보호 조건이며 공식 범주 개수를 뜻하지 않습니다.

전체 실행은 원본 Train만 메모리에 모아 전처리 기준을 학습합니다. 검증·변환은 `chunksize`만큼 나누어 처리하고, 출력은 float32 디스크 배열로 저장합니다.

메모리가 부족하면 먼저 표본 모드를 사용합니다. `chunksize`를 줄이면 읽기·변환 중 메모리는 줄지만 Train 전체를 학습할 때 필요한 메모리는 줄지 않습니다. 오버샘플링은 이 단계에서 적용하지 않습니다.

## 출력 파일

실제 파일을 검증한 뒤에만 통계·표·그래프를 생성합니다. `--audit-only`는 집계와 실행 기록만 만들며 변환기를 학습하지 않습니다.

SHA-256은 파일 내용이 같은지 확인할 수 있는 해시값입니다. 원본 파일의 해시를 실행 기록에 남깁니다.

| 위치 | 내용 |
|---|---|
| `results/phase1/<run>/schema.csv` | 실제 데이터 타입과 원래 결측·특수 결측·변환 후 결측 수를 저장합니다. |
| `results/phase1/<run>/monthly.csv`, `splits.csv` | 월별·분할별 행 수와 사기 비율을 저장합니다. |
| `results/phase1/<run>/numeric_by_month.csv`, `categories_by_month.csv` | 특성별 월별 수치 요약과 범주 분포를 저장합니다. |
| `results/phase1/<run>/monthly.png`, `missingness.png`, `monthly_mean_shift.png` | 월별 행 수·사기 비율, 결측률, 월별 평균 변화를 그립니다. |
| `results/phase1/<run>/feature_mapping.json` | 변환 열의 인덱스와 원래 특성의 대응을 저장합니다. |
| `results/phase1/<run>/manifest.json` | 원본 SHA-256, 설정, 분할, 패키지, 장치와 처리 시간을 저장합니다. |
| `data/processed/<run>/*_X.npy`, `*_y.npy`, `*_row_ids.npy` | 분할별 입력·정답과 원본 행 위치를 저장하며 Git에서 제외합니다. |
| `data/processed/<run>/preprocessor.joblib` | Train에서 학습한 변환기를 저장하며 Git에서 제외합니다. |

전처리 폴더에도 같은 실행에서 생성한 manifest와 특성 대응 정보를 저장합니다. JSON 사본은 코드가 함께 생성하므로 손으로 맞추지 않습니다. `row_ids`는 CSV의 행 위치이며 신청자 식별자가 아닙니다. 외부에서 받은 joblib 파일은 로드하지 않고, 직접 생성한 변환기만 사용합니다.

공식 Base 전체에서 확인한 스키마, 분할 통계, 특성 대응, 검증 결과와 그래프는 [검증 결과 사본](results/reference/baf-base/)에 보관합니다. 실행 때 생성하는 출력과 로컬 검토 기록은 Git 추적에서 제외합니다.

월별 평균 변화 그림의 z-score는 각 특성의 월별 평균을 비교하기 위한 EDA 값입니다. EDA는 탐색적 데이터 분석을 뜻합니다. 이 그림의 계산값은 전처리 변환기에 전달하지 않습니다. Test의 분포·라벨 집계는 데이터 유효성 점검에만 사용하고 모델·전처리 방법·임계값을 고르는 근거로 사용하지 않습니다.

## 테스트

테스트는 작은 합성 데이터를 사용하며 실제 BAF의 통계나 학습 성능을 측정하지 않습니다.

```bash
python -m pytest -q
python -m pip check
```

시간 분할의 중복·누락, Train 외 데이터의 fit 거부, Train 중앙값·스케일링, 미지 범주의 처리, 정답·시간 특성 제외, 특수 결측 규칙, 특성 대응과 직렬화, CSV 인덱스, 오류 시 출력 중단을 검증합니다. 전체 전처리 실행과 작은 표본 실행도 임시 폴더에서 확인합니다.

## 로컬과 학교 GPU 서버

환경 확인 명령은 실행한 컴퓨터만 검사합니다. SSH로 서버에 접속하거나 인증·시스템 설정을 바꾸지 않습니다.

```bash
baf-env --output results/environment/local.json
# 서버에 공식 접속한 뒤 같은 명령을 실행할 수 있습니다.
baf-env --output results/environment/server.json
```

PyTorch가 설치되어 있으면 CUDA, MPS, CPU 순서로 장치를 선택합니다. 사용할 수 없는 장치를 명시하면 오류로 처리합니다. PyTorch가 없으면 자동 장치 선택값을 null로 기록합니다. 데이터 전처리는 CPU에서 실행하므로 환경 기록의 선택 장치가 전처리의 GPU 실행을 뜻하지는 않습니다.

학교 서버의 운영체제·Python·GPU 모델·VRAM·드라이버·PyTorch CUDA·저장 공간은 실제 서버에서 확인합니다. `baf-env`는 패키지와 장치, 여유 공간을 기록하며 `nvidia-smi`가 있으면 GPU와 드라이버 조회 결과도 기록합니다. 기존 가상환경과 학교의 자원 사용 규칙은 별도로 확인해야 합니다. 서버 접속이나 환경 변경은 변경 범위를 설명하고 확인받은 뒤 진행합니다.

```bash
nvidia-smi
python --version
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
df -h .
```

서버 사양과 정책을 확인한 뒤 전체 데이터 학습과 GPU 메모리에 맞는 batch size를 정합니다. CUDA 혼합 정밀도는 환경 확인 뒤 `configs/b0.toml`의 `mixed_precision`을 `fp16`으로 설정해 사용합니다.

## MLP 학습

PyTorch 환경에서 `baf-data`로 만든 Train·Validation 배열을 사용합니다. `configs/b0.toml`에서 모델, 시드, 학습 설정과 체크포인트 위치를 관리합니다. 로컬에서는 `--smoke`로 각 분할의 클래스 비율에 맞춰 작은 표본을 추출할 수 있습니다.

```bash
python -m pip install '.[torch]'
baf-train --run-name local-first --smoke --batch-size 256 --epochs 2
baf-train --run-name local-resumed --smoke --batch-size 256 --epochs 3 \
  --resume checkpoints/b0/local-first/last.pt
```

`--epochs`는 재개 전 학습을 포함한 총 epoch 수입니다. 재개 명령에는 새 실행 이름과 같은 데이터·학습 설정을 사용합니다. `metrics.json`에 epoch별 손실·Validation AP·장치·시간을 기록합니다. `best.pt`는 Validation AP가 가장 높았던 모델을, `last.pt`는 마지막 epoch의 모델과 복구에 필요한 학습 상태를 저장합니다.

E1은 B0와 같은 MLP에 Weighted BCE를 적용합니다. 실제 학습에 쓰는 Train 행에서 정상 건수를 사기 건수로 나눠 `pos_weight`를 계산하고 `metrics.json`에 기록합니다. E2는 고정된 클래스 가중치 없이 Focal Loss를 적용합니다. `configs/e2.toml`의 `focal_gamma`는 모델이 쉽게 맞히는 사례의 손실을 얼마나 줄일지 정합니다. [Focal Loss 논문](https://arxiv.org/abs/1708.02002)

세 실험은 Validation 손실을 가중치 없는 BCE로 계산하며, 체크포인트는 Validation AP로 선택합니다. CUDA에서 B0를 배치 크기 256으로 학습했다면, 같은 장치와 배치 크기로 E1과 E2를 실행합니다.

```bash
baf-train --config configs/e1.toml --device cuda --batch-size 256 --run-name school-e1-seed42
baf-train --config configs/e2.toml --device cuda --batch-size 256 --run-name school-e2-seed42
```

E2에서 감마의 영향을 확인하려면 `--focal-gamma`를 사용합니다. 여러 시드로 반복할 때는 `--seed`에 시드를 지정합니다. 실행 이름에 감마와 시드를 넣으면 결과 폴더를 찾기 쉽고, 실제 적용된 값은 `metrics.json`에서 확인할 수 있습니다.

```bash
baf-train --config configs/e2.toml --device cuda --batch-size 256 \
  --focal-gamma 0.5 --seed 43 --run-name school-e2-gamma0.5-seed43
```

[MLP 손실함수 비교 기록](results/reference/mlp-loss-comparison/report.md)에서 학교 GPU의 세 시드 결과, 감마 탐색, 로컬 검증 기록을 확인할 수 있습니다.

E3는 너비 75의 Residual MLP에 Weighted BCE를 적용합니다. Residual 블록은 입력을 두 선형층으로 변환한 결과에 원래 입력을 더합니다. E1의 MLP와 파라미터 수를 비슷하게 맞췄으며, 실제 개수는 `metrics.json`의 `trainable_parameters`에 기록합니다. E1과 같은 시드 및 배치 크기로 반복합니다.

```bash
baf-train --config configs/e3.toml --device cuda --batch-size 256 \
  --seed 42 --run-name school-e3-seed42
```

[MLP 구조 비교 기록](results/reference/mlp-architecture-comparison/report.md)에서 E1·E3의 세 시드 Validation AP, 파라미터 수, 학교 GPU 학습 시간을 확인할 수 있습니다.

## 연구 순서

먼저 단일 시드로 학습 동작을 확인하고 주요 실험은 최소 세 시드로 반복합니다.

| 실험 | 모델 | 학습 방법 | 범위 |
|---|---|---|---|
| B0 | MLP | BCE | 필수 실험으로 진행합니다. |
| E1 | MLP | Weighted BCE | 필수 실험으로 진행합니다. |
| E2 | MLP | Focal Loss | 필수 실험으로 진행합니다. |
| E3 | Residual MLP | Validation에서 선정한 손실함수 | 필수 실험으로 진행합니다. |
| E4 | MLP | BCE와 Train 전용 Random Oversampling | 시간과 자원이 허용되면 추가합니다. |
| E5 | FT-Transformer | 선정한 손실함수 | 시간과 자원이 허용되면 추가합니다. |

MLP와 Residual MLP의 구조 효과는 같은 손실함수·분할·시드·튜닝 예산에서 비교합니다. Validation AP를 기준으로 체크포인트를 선정하고, 설정한 epoch 동안 개선이 없으면 Early Stopping을 적용합니다. AP는 불균형 데이터에서 Precision과 Recall의 관계를 요약하는 Average Precision입니다.

FPR은 정상 신청 중 사기로 잘못 판정한 비율입니다. 평가에서는 Validation에서 FPR 1% 이하의 임계값을 정하고 Test에 고정 적용합니다. Test의 실제 FPR과 Recall을 함께 보고하며 AP·Precision·Recall·F1·ROC-AUC·혼동행렬·PR 곡선·월별 성능과 학습 시간을 비교합니다.

학습이 안정화되면 선정 모델에 Captum Integrated Gradients를 적용합니다. logit 또는 확률 중 설명 대상을 명시하고 baseline을 기록하며, 범주형 기여도를 원래 특성으로 합칩니다. 전체 중요도와 TP·TN·FP·FN 사례를 분석하고 인과적 원인으로 해석하지 않습니다. SHAP은 필요할 때 추가합니다.

## 코드 위치

`src/baf/schema.py`는 스키마와 특수 결측 규칙을 정의합니다. `data.py`는 CSV 읽기와 시간 분할을, `audit.py`는 집계를, `preprocessing.py`는 Train 전용 변환을 담당합니다. `cli.py`는 데이터 실행 명령을 제공하고 `runtime.py`는 현재 컴퓨터의 환경을 확인합니다. `scripts/`는 같은 기능의 Python 실행 진입점이며, `tests/`는 핵심 계약을 검증합니다.
