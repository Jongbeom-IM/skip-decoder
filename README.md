#YOLOv7

## Skip-decoder 구조
![image](https://github.com/user-attachments/assets/91050ef4-47cd-453c-a48c-365eacf836ff)

### 튜토리얼
1. [COCO dataset](https://cocodataset.org/#download) 을 로컬에 저장해주세요
    *  본 실험은 COCO 2017 Train과 COCO 2017 Val을 사용했습니다.
    *  YOLO 모델의 데이터 레이블 형식은 COCO에서 제공하는 레이블과 다릅니다. 브랜치에서 COCO dataset의 레이블을 저장해주세요.
    *  data/coco_full.yaml 에 데이터셋 경로를 수정하세요.
2. 디코더와 커스텀 레이어 구조는 models/common.py 에서 확인할 수 있습니다.
    * 커스텀 레이어 추가시, 레이어 정보(Class)를 models/yolo.py의 ***parse_model()*** 함수에 추가해 주세요.
    * 디코더 위치 정보를 parse_model()의 decoder 인자로 넘깁니다. ***Class decoder()*** 에서 어떤 디코더를 선택하는지 확인해주세요. 
3. 모델 디버깅 에러 프로파일링
    * 평소에는 주석 처리된 parse tensor shape 부분을 주석 해제하여 레이어 출력이 어디까지 되었는지 확인하세요.
    * 보통, ***parse_model()*** 에서 레이어 추론에 필요한 인자를 제대로 넘기지 않아서 그렇습니다.
4. 학습 예시
    * docker start skip-decoder (YOLOv7을 위한 도커 환경을 설치해주세요.)
    * python train.py --batch-size 16 --cfg cfg/deploy/e6e_SD1.yaml --epochs 200 --name yolov7_SD1
