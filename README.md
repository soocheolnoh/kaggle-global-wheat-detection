# Final Review (2020-08-21)
* 최종 결과: 86/2245, top 4% 달성 (은메달)
  * public 216/2245에서 big shakeup
* big shakeup의 이유?
  * 다른 사람들이 무리하게 PL을 진행하여 public을 올리지 않았을까?

| Model | Public (Before Corrected) | Public (After Corrected) | Private (After Corrected) |
| :--- | :---: | :---: | :---:|
| EffDet-d5, 1024                       | < 0.74 | 0.7396 | 0.6118 |
| EffDet-d5, 1024 + PL 1 round 5 epochs | 0.7415 | 0.7254 | 0.6540 |

* PL은 어떻게 진행했어야 했을까?
  * 5 epochs도 많은 것 같음
  * training data + 2 * test data를 이용하여 1 epoch 느낌으로, 아주 조금 tuning 하는 정도로 해야 성능이 나올 것 같음
  * learning rate의 잘못된 사용?
    * 나는 실제 model train/valid 구현 위치에서 load해서 하였는데, HeadNet 불러오는 것에서 load했어야
  * 똑같이 valid set을 이용하여 overfit하지 않았는지 확인했어야 함
    * test set에서도 valid set을 구분했어야, 결국 valid set = train set / 5 + test set / 5
    * OOF를 진행했다면, valid set을 어떻게 사용했어야 했을까?: 하나의 fold만 사용할 수밖에 없을 것 같음
* TTA 단계에서의 cutout
  * 4개의 꼭지점에서의 cutout을 사용해 TTA 40을 사용하는 방법은 꽤 효과가 있어 보임
  * 해당 model은 private 39/2245로 top 2%에 해당
  * 경계 조건을 0.5로 준 것인데도, 실제 test image 상에서 오류가 있었으므로, 넉넉하게 2를 준다면, 더 상향될 것으로 보임
  
| Model | Public (Before Corrected) | Public (After Corrected) | Private (After Corrected) |
| :--- | :---: | :---: | :---:|
| EffDet-d5, 1024 + TTA40 | < 0.74 | 0.7342 | 0.6618 |

* 전처리
  * 작은 bbox를 없애는 전처리는 실제 수정되기 전 public에서는 성능이 떨어졌지만, 수정 후라면 상향 됐을 수도 있음
  * 이외에 다른 방법이 있었을지 고민해볼 필요가 있음

# Review after Deadline (2020-08-06)
* 목표했던 10%는 달성하기 힘들어 보임 (한 10-20% 예상됨)
* 사용했던 모델은 yolov5, efficientdet
  * yolov5를 처음에 사용했으나, license 문제로 out
  * yolov5에서 바로 efficientdet으로 진행했어야 되는데, 우물쭈물하면서 날린 시간이 컸음
* efficientdet의 구조는 논문을 통해 대략적으로 알게됨
  * backbone인 efficientnet은 나중에 한 번 읽어보자
  * 최초 yolo model부터 yolov4까지의 변천사도 확인해 보자
* kaggle notebook 사용법, colab 사용법 등을 배움
* pytorch 상에서 image augmetation 하는 법을 배움
  * 단순히 training set의 개수를 늘리는 줄 알았지만, albumentation을 사용
  * albumentation이 LB값에 엄청나게 큰 영향을 줌
* training 단계에서의 전처리
  * 대략적으로 구현은 하였지만, 실제 public LB에서 부정적 영향이라 제외함
  * 시간이 좀 더 있었다면, 포함시켜 training 하고 같이 제출할 듯
  * 실제 private data에는 bbox가 전처리 되어 있다고 하니, 더 나은 결과가 나왔을 수도 있음
* cutout tuning은 성공적이었음
* boxcut augmentation은 결국 사용하지 않음
  * 개인적인 생각에, cutout을 제외시키고, boxcut을 더 잘 적용시키면 됐었을 것 같음
* TTA에서 image 확대/축소 추가: 사용하지 않음
  * 어떻게 하더라도, 제외시킨 것보다 LB가 높아지지 않음 (saturation 되는 느낌)
  * 다른 방법이 있지 않았을까?
* 제출할 때의 error
  * training data를 사용하지 말자
* test data를 통해서 제출하는 거지만, training data를 통해서 한 번 검증을 해보자

# Leaderboard
* img_size=512

| LB | Loss | cutmix | cutout | album | rotate | boxcut | TTA | temp |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 0.6361 | 0.3791 | O |         |           |   |       |            |
| 0.6451 | 0.3869 |   |         |           |   |       |            |
| 0.7023 | 0.3668 |   |         | O         | 4 |       |            |
| 0.7045 | 0.3644 | O |         | O         | 4 | p=1.0 |            |
| 0.7077 | 0.3637 | O | s64n64  | O         | 4 | p=1.0 |            |
| 0.7094 | 0.3629 |   | s64n32  | O         | 4 |       |            |
| 0.7100 | 0.3608 | O | s64n32  | O         | 4 |       |            |
| 0.7110 | 0.3655 |   | s128n8  | O         | 4 |       |            |
| 0.7174 | 0.3610 | O | s128n8  | O         | 4 |       |            |
| 0.7252 | 0.3627 | O | s32n256 | O         | 8 |       | rot8       |
| 0.7255 | 0.3604 | O | s64n64  | O         | 4 |       |            |
| 0.7271 | 0.3630 | O | s64n64  | O         | 8 | p=0.5 | rot8+color |
| 0.7279 | 0.3702 | O | s64n64  | O+gray1.0 | 8 |       | rot8       |
| 0.7300 | 0.3600 | O | s64n64  | O         | 4 | p=0.5 | rot8       |
| 0.7313 | 0.3654 | O | s64n64  | O+gray0.0 | 8 |       | rot8       |
| 0.7334 | 0.3630 | O | s64n64  | O         | 8 | p=0.5 | rot8       |
| 0.7343 | 0.3636 | O | s64n64  | O         | 8 |       | rot8       |
| 0.7344 | 0.3604 | O | s64n64  | O         | 4 |       | rot8       |
| 0.7348 | 0.3617 | O | s32n128 | O         | 8 |       | rot8       | corrected |
| 0.7360 | 0.3636 | O | s64n64  | O         | 8 |       | rot8       | corrected |

* img_size=1024

| LB | Loss | cutmix | cutout | album | img |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.7229 | 0.3889 | O | s128n8 | O | 1024 |

# TODO
* [x] random cutout 방법 ([pdf](https://arxiv.org/pdf/1708.04896.pdf))
* [ ] <del>Auto Augmentation ([pdf](https://arxiv.org/pdf/1905.00397.pdf))</del>
* [x] Test Time Augmentation
* [x] Psheudo Labeling
* [x] Albumentation Tuning
* [ ] <del>Custum Image Augmentation: Bounding Box만 추출하여 이미지를 어떻게 생성시킬 수 있지 않을까?</del>
* [ ] <del>Image Mix-up Augmentation</del>
* [ ] <del>K-fold Ensemble (이건 제일 마지막 img_size 1024로 해야 할 듯, training 시간만 12*5시간 예상)</del>
* [ ] <del>Training 단계에서의 전처리: fold id가 바뀔 수 있으므로, 후순위</del>

# 2020-08-04
* submit
  * round1-epoch5
  * round2-epoch5 (한 번 kaggle error로 실패)
  * round3-epoch5

# 2020-08-03
* submit
* LB 0.7157: cutout(s64n64)+TTA16(reduce) 효과 없음
  * LB 0.7314: cutout(s32n64)+TTA8 효과 없음
  * LB 0.7293: prebbox+cutout(s32n64) 효과 없음
  * 2개 중 한 개를 centercrop 개선으로 생각해보자
* TTA
  * cropreduce는 작게 보는 것이기 때문에, 효과 없을 것 같았음
  * centercrop을 좀 더 개선할 수는 없을까?
* prebbox
  * 작은 bbox은 실제 해당 위치에 wheat가 있지만, 짤렸기 때문이 아닐까?
  * 큰 bbox들만 수정하고, 해당 위치에 추가 wheat를 그릴 수 있으면 좋을듯
  * 개선 방법이 어려울듯
* training
* cutout(s64n64)-1024-fold0으로 진행중
  * 25+25 epochs 가능할까? 시간 확인해보자

# 2020-08-02
* submit
  * 기본weight + 경계제거(0.5offset) center-cut TTA
  * 다시 submit error, bbox의 경계 조건을 명확히 하여 다시 제출
    * 경계 조건 명확히 해도 error, 다른 이유가 있나?
  * training data를 이용하여 해결
    * squeeze()를 추가했던 것이 원인
  * LB 0.7356: centercut TTA 추가
    * boosting 될 줄 알았으나 오히려 살짝 떨어짐
  * LB 0.7349: centercut 2개 TTA, 총 TTA24
* centercut
  * 결국 boundary에서 loss가 발생하기 때문에 더 좋아지지 않는듯
  * 확대보단 축소를 해야 더 좋아질까?
* 다음 submit
  * 기본 weight로 축소(450) 제출
  * cutout-32n64 / prebox 제출
* 현재 training
* cutout-32n64-prebox, 1024, epoch25로 traning중

# 2020-08-01
* cutout size/2, num으로 진행
* colab에서 pytorch의 버전업으로 effdet 코드에서 error
  * effdet github에서 최신 버전으로 교체
* inference에서 resize test중
  * 알 수 없는 원인으로 submission에서 error 발생
  * 처음을 그냥
512로 resize하고 그 상태에서 cutout 구현하자
* cutout 구현 후 submit
  * centercutout으로 구현
  * LB 0.6721: boundary 부분을 제거해야 할 듯
  * 경계에서 0-1 부분을 제거 (0-0.5로 해도 될 듯? 다시 한 번 체크)
* 내일 submit 해볼 것
  * 기본 weight로 경계 제거된 center-cutout TTA 제출
  * cutout-s32n64 제출 (TTA는 위 결과 보고 best 사용)
  * cutout-s32n64-prebbox(10-512) 제출

# 2020-07-31
* gray 1%는 약 0.5% boosting인 걸로 나타남
* gray 100%로 training 진행중
  * 더 안좋아짐
* cutout size/2, num*4 진행
  * 결과 좋지 않음
  * 다시 num*2로 진행해보자
* training 및 test에서 xyxy, xywh등으로 변환할 때 오류 수정
  * 생각보다 영향이 컸던 것 같음, 0.2~0.3% boosting

# 2020-07-30
* bbox + rot8 성능 더 안좋음
  * loss값은 낮아졌지만, rot4에 비해 0.5% 정도의 boosting
  * bbox 제외시키고 다시 rot8만 넣어 진행하면 갱신 가능
* TTA에 color shift 적용
  * 예상대로 성능이 더 좋지 않음
  * training 단계에서, color shift를 적용시키면 더 좋아지지 않을까?
  * hue/brightness/gray 등의 영향도를 살펴보자
* 게시판을 보다 RAdam이라는 optimizer가 성능이 좋다는 이야기가 있음
  * 현재 AdamW 사용중인데, 해당 optimizer로 바꿔보는 것도 좋은 시도
* bbox cut은 loss를 낮추긴 하는데, 실제 LB값은 더 안좋아짐

# 2020-07-29
* 기본적인 rotation/flip 적용하여 8가지 TTA 적용
  * 추가적인 TTA 생각: 현재로써는 color shift
  * color shift는 training 단계에서도 하지 않기 때문에 영향이 클 것 같지 않음
* bbox cutout을 확률적으로 적용하니 loss 조금 낮아짐
  * TTA와 합쳐 실제 inference 진행
  * 실제 LB는 안좋아짐
  * bbox를 조금 잘라내서 그런 것일까?
* 추가로 random rotation 적용시켜 8가지 image 모두 같은 확률이도록 albumentation
  * training 진행중
* 그 다음 해볼 만한것
  * image cutout에 0 대신 random값 넣기
  * bbox cutout에 while문 제거: 자연히 관련 parameter 조절
  * TTA에 image crop 적용: 똑같이 800x800으로 crop하여 적용해야하지 않을까?
  
# 2020-07-28
* cutout 제외 boxcut training 결과
  * 나쁘진 않아보이나 best는 아님
  * cutout 포함하여 다시 training
* opt_parameter 관련 코드 주석이 영향이 있을까?
  * 현재 training 하는 거 이후, 주석 해제 후 다시 해보자
  * 영향 없어 보임
* bbox cutout이 꽤 성능 상향해줄줄 알았는데, 오히려 햐향
  * 무조건 적용시켜서 그러지 않을까?
  * cutout은 흑색, bbox cutout은 랜덤이라 그러지 않을까?
  * bbox cutout 흑색으로 다시 training
* 그 후, 확률을 각 box에 적용하는 것이 아니라, bbox cutout에 적용시키자
  * 속도 느린 bbox보다 while문 없앤 bbox 적용

# 2020-07-27
* 현재 setting을 gradient accumulation 적용
  * epoch=50으로 training하고 결과 확인: 덜 training 되어 보임
  * training에서 resume 구현 기능으로 추가 training 할 수 있을듯
  * 현재 resume 기능 테스트 진행: 50+50 epoch, loss는 좋진 않음
  * 사용해보았으나 좋아보이진 않음, 쓰진 않을 듯
  * resume 기능은 잘 작동되는 것으로 보임
* training에서 x2, y2를 계산할 때, 이는 포함되지 않음
  * x2 = x1+w 식으로 계산했기 때문에 boundary에 들어가게 됨
  * \[x1, x2), \[y1, y2) 이런 식
* 1차적으로 bbox random cutout 구현
  * 기본 파라미터 사용 (p=0.5, size는 0.02~0.2)
  * mixup-albumentation-bbox cutout 순으로 진행
  * 추후 전체 이미지에 대한 cutout 추가 예정

# 2020-07-26
* cutout size 줄인 것의 loss는 더 좋지만, LB는 더 나빠짐
* bbox 안의 cutout이 상대적으로 적어진 게 원인일까?
* 일단 개수 2배(s64-n64)로 설정하고 다시 한 번 확인해 보자
  * LB 0.7255로 엄청 좋아짐
* 1순위: custom cutout
* 추가로 gradient accumulation을 적용해서 해 보자
  * batch_size를 간접적으로 늘려주는 skill인 것 같은데, 정확히 이해가 안 됨
  * 먼저 epoch=2로 training 시간 테스트

# 2020-07-25
* inference kernal에 혹시 모르니까 seed 설정 함수 추가
* cutout 제거한 것 성능 평가
* 1순위: custom cutout
* 현재 cutout size 1/2로 하고 개수 4배로 training 진행중: 더 나아보임
* cutmix 포함시켜서 다시 training 시키고, 앞으로는 cutmix와 같이 해야 할 듯

# 2020-07-24
* 완전 Baseline (no album & cutmix) 성능: 0.6451
  * 놀랍게도 cutmix만 적용한 것보다 성능이 좋게 나옴
* cutmix의 성능
  * 구현이 원 논문과 비교해서 좀 다르지 않을까?
  * cutmix는 단독 사용보다는 다른 augmentation과 같이 써야 성능이 나온다?
* 먼저 cutout의 성능 개선을 해보자
  * 현재 cutout의 성능 확인 필요: 기본 album 해보고, cutout 제거 한 뒤 해보자
  * random cutout: 단순 구현은 쉬워보이지만, bbox끼리 겹치는 부분, 확률을 어떻게 적용해야 할까는 생각해야

# 2020-07-23
* training kernal 틀 완성
* apex 사용 추가
* 512보단 1024가 성능 좋음
* 일단 기본적인 base model은 fold0의 img_size 512, batch 4, epoch 50로 진행
* 먼저, albumentation과 cutmix의 성능을 알아보기 위해 없이 테스트 필요
* LB Scores
  * base 512 성능: 0.7174
  * only cutmix: 0.6361 (overfit하는 것 같아서 중간에 멈춤)
* 추가로 no album & no cutmix 진행 필요

# Before 2020-07-23
* [yolov5](https://github.com/ultralytics/yolov5) 시도
  * submission not found error: train 폴더 및 csv 참조가 원인
  * yolov5 기본 parameter로 5 fold model training
  * 5 fold model을 이용하여 ensemble 진행 ([weighted boxes fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion))
  * 처음엔 kaggle kernal 이용 → GPU 시간 제한 30시간 / 1 week
  * kaggle kernal에서 google colab으로 이동
  * google colab에서의 drive 읽기 속도는 매우 느리므로, 먼저 drive에서 파일을 colab에 복사해야 함
  * pytorch를 사용할 경우, apex를 쓰면 속도 향상이 있을 수 있음 (보통 'O1'을 많이 사용)
  * colab은 시도 때도 없이 runtime 종료가 발생하므로, 간단한 js 함수 하나를 이용해 이를 방지
  * yolov5의 license는 GPL로 kaggle competition license에 맞지 않아 퇴출
* efficientdet 시도: pytorch 상에서 구현한 코드 중 하나인 [yet another efficientdet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) 이용
* yolov5와 비슷하게 train.py를 이용해 custom data를 training 해보려고 시도
  * training은 잘 되나, apex 지원이 없음
  * test 및 valid도 뭔가 진행하기 어려워 보였음
* 다른 코드 efficientdet 시도: rwightman의 [efficientdet](https://github.com/rwightman/efficientdet-pytorch) 시도
* kaggle에 해당 코드 안의 efficientdet class를 이용한 pipeline notebook이 있어서 이를 이용하기로 함
  * 이 notebook은 train.py를 이용하는 것이 아니라 class를 이용하여 pytorch 상에서 구현 (forward-backward 등)
  * 처음에는 code 이해가 어려워 바로 fork 후 실행 먼저 진행
  * image augmentation library로써 albumentation을 사용함
  * forward-backward-validation 과정을 거침
  * 각 과정에서 loss와 scheduler를 이용하여 lr update
