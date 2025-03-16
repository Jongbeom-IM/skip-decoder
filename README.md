## COCO labling
초기에 COCO lable을 저장하면 .txt 가 아닌, annotation파일이 제공됩니다.

**lable 폴더를 저장해 그대로 사용하셔도 무관합니다.**

본 소스코드는 annotation 정보에서 YOLO 의 segmentation에 필요한 정보로 재가공하는 기능을 합니다.

labling.py → normalization.py 순서로 코드를 실행하되, 다음 지시 사항을 따라 주세요.
1. labling.py 실행 후 label/에 tain2017과 val2017이 생성됩니다. 이를 old_tain2017과 old_val2017로 변경해주세요.
2. 이를 normalization.py를 사용해 YOLO에 맞는 segmentation boundary로 수정합니다.
3. recategory.py는 데이터셋의 클래스(cat, bycle, ....)을 숫자가 아닌 영문 클래스로 바꿉니다. 상황에 따라 적절하게 사용해주세요.
