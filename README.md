# Collaborative Intelligence with Skip-Decoder for Balanced Edge Power and Cloud Cost in Object Detection
## Introduction 
![image](https://github.com/user-attachments/assets/27b8480a-baf0-4d2e-88e0-26a736d40410)

 - 높은 처리량을 요구하는 딥러닝 모델을 리소스가 낮은 Edge device에서 실행하기 위한 추론 Framework
 - Edge-Cloud 환경으로 분산 추론하여 Edge device의 처리량을 절약

### Skip connection 아키텍처에서 발생하는 분산 추론 이슈
 - Nvidia Jetson AGX Xavier, TX2(Edge)와 RTX 4090 server(Cloud)에서 분할 지점 별 총 추론 시간 프로파일링
   + 분산 추론이 없는게 더 빠름
 - Skip connection 아키텍처의 특성으로 인해 다중 Feature-map을 전송해야 하는 문제가 발생
   + 분할 지점 별 Featuer map 전송량이 추론 시간의 가장 큰 의존성을 보임
![image](https://github.com/user-attachments/assets/accaf2c3-1c94-4b31-b3cf-d0d9cb2dce01)


## Methodology
### Ours(Skip-decoder)
![image](https://github.com/user-attachments/assets/9ef511ed-6e5e-4a19-a632-813e10dcb860)
- 에지 디바이스의 소모 전력, 클라우드의 연산 비용을 종합적으로 고려한 Real-time 환경의 분산 추론 framework 설계
- Feature map 전송 비용을 최소화 하는 경량화된 Feature reconstructor 설계 → 클라우드 연산 비용 절감
- Skip-decoder는 모델 분할 직전의 레이어를 Transposed convolution으로 변환-복제하여 역순으로 배치

### 개선효과
![image](https://github.com/user-attachments/assets/72fadcda-c7f7-4205-a8ba-ed7e7c6f3e17)


## Code review
YOLOv5/YOLOv7 가 있습니다. 각 브랜치를 확인해 주세요.

**용량 문제로 실험 데이터(학습된 모델)은 클라우드 링크로 걸어두겠습니다**

[yolov5](https://drive.google.com/drive/folders/16-N7wI42LfDLAiTy9sJ8t5E3oX6hRWz2?usp=drive_link)
[yolov7](https://drive.google.com/drive/folders/1e5gszbORpQkKYbot4nGlFafJNdXuOS2n?usp=drive_link)


## Contact
**Phone**: 010-5649-4952

**Email**: jongbeom.im@g.skku.edu

**Paper**: https://ieeexplore.ieee.org/document/10773834
