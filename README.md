# Level1-Image-Classification-CV-14

# 😄 팀 소개

<table align="center">
    <tr align="center">
        <td><img src="https://github.com/user-attachments/assets/3560856a-8cac-4079-8494-4f1bf13d0eb5" width="125"></td>
        <td><img src="https://github.com/user-attachments/assets/3560856a-8cac-4079-8494-4f1bf13d0eb5" width="125"></td>
        <td><img src="https://github.com/user-attachments/assets/3560856a-8cac-4079-8494-4f1bf13d0eb5" width="125"></td>
        <td><img src="https://github.com/user-attachments/assets/3560856a-8cac-4079-8494-4f1bf13d0eb5" width="125"></td>
        <td><img src="https://github.com/user-attachments/assets/fdce3bf1-4dd2-44c9-b4db-1599c4d3826d" width="125" height="125"></td>
        <td><img src="https://github.com/user-attachments/assets/3560856a-8cac-4079-8494-4f1bf13d0eb5" width="125"></td>
    </tr>
    <tr align="center">
        <td><a href="https://github.com/kimgeonsu" target="_blank">김건수</a></td>
        <td><a href="https://github.com/202250274" target="_blank">박진영</a></td>
        <td><a href="https://github.com/oweixx" target="_blank">방민혁</a></td>
        <td><a href="https://github.com/lkl4502" target="_blank">오홍석</a></td>
        <td><a href="https://github.com/Soy17" target="_blank">이소영</a></td>
        <td><a href="https://github.com/yejin-s9" target="_blank">이예진</a></td>
    </tr>
</table>

<br/>

# 📋 Project Overview
Sketch이미지 분류 경진대회는 주어진 데이터를 활용하여 모델을 제작하고 어떤 객체를 나타내는지 분류하는 대회입니다.

Computer Vision에서는 다양한 형태의 이미지 데이터가 활용되고 있습니다. 이 중, 비정형 데이터의 정확한 인식과 분류는 여전히 해결해야 할 주요 과제로 자리잡고 있습니다. 특히 사진과 같은 일반 이미지 데이터에 기반하여 발전을 이루어나아가고 있습니다.

하지만 일상의 사진과 다르게 스케치는 인간의 상상력과 개념 이해를 반영하는 추상적이고 단순화된 형태의 이미지입니다. 이러한 스케치 데이터는 색상, 질감, 세부적인 형태가 비교적 결여되어 있으며, 대신에 기본적인 형태와 구조에 초점을 맞춥니다. 이는 스케치가 실제 객체의 본질적 특징을 간결하게 표현하는데에 중점을 두고 있다는 점을 보여줍니다.

이러한 스케치 데이터의 특성을 이해하고 스케치 이미지를 통해 모델이 객체의 기본적인 형태와 구조를 학습하고 인식하도록 함으로써, 일반적인 이미지 데이터와의 차이점을 이해하고 또 다른 관점에 대한 모델 개발 역량을 높이는데에 초점을 두었습니다. 이를 통해 실제 세계의 복잡하고 다양한 이미지 데이터에 대한 창의적인 접근방법과 처리 능력을 높일 수 있습니다. 또한, 스케치 데이터를 활용하는 인공지능 모델은 디지털 예술, 게임 개발, 교육 콘텐츠 생성 등 다양한 분야에서 응용될 수 있습니다.



- 입력
  - **class_name(각 이미지별 폴더명), image_path, target(예측해야할 class)**
- 출력
  - **id, image_path, target**

<br/>

# 🗃️ Dataset

- 전체 이미지
  - **9754 images**
  - train
    - **15021 (500classes * (29~31))images**
  - test
    - **10014 images**
- 클래스 수
  - **500 classes**
