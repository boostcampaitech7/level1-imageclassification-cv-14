import os
import pandas as pd
from diffusers import StableDiffusionPipeline
from PIL import Image
from multiprocessing import Pool, Manager
import time


class StableDiffusionImageGenerator:
    def __init__(self, model_name="CompVis/stable-diffusion-v1-4", device="cuda"):
        # Stable Diffusion 모델 로드
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name).to(device)

    # 스케치 이미지를 생성하는 함수
    def generate_sketch(self, prompt: str, num_images: int):
        images = self.pipe(prompt, num_inference_steps=30, num_images_per_prompt=num_images).images  # 이미지 생성
        return images

    # 이미지를 저장하고 CSV 파일에 추가하는 함수
    def save_and_update_csv(self, lock, class_name, class_target, text, output_dir, num_images, csv_file):
        # 이미지 저장 경로 생성
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # 이미지 생성 및 저장
        for i in range(num_images):
            # 프롬프트에 기반한 스케치 생성
            prompt = f"a sketch of {text}"  # 클래스 이름에 맞는 프롬프트
            images = self.generate_sketch(prompt, num_images=1)

            # 이미지 파일명 생성 및 저장
            image_file = f"{class_name}_generated_{i}.JPEG"
            image_path = os.path.join(class_dir, image_file)
            images[0].save(image_path)  # 이미지 저장

            # CSV 파일에 기록하기 위한 잠금
            with lock:
                # CSV 파일 읽기
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                else:
                    df = pd.DataFrame(columns=["class_name", "image_path", "target"])

                # CSV 데이터프레임에 정보 추가
                new_row = {"class_name": class_name, "image_path": os.path.join(class_name, image_file), "target": class_target}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                # CSV 파일 저장
                df.to_csv(csv_file, index=False)

    # 모든 클래스에 대해 반복적으로 이미지 생성 및 CSV 업데이트
    def generate_images_for_classes(self, class_info_csv, output_dir, num_images_per_class, csv_file):
        # 클래스 정보가 담긴 CSV 파일을 읽음
        df = pd.read_csv(class_info_csv)
        manager = Manager()
        lock = manager.Lock()

        start_time = time.time()

        # 병렬로 각 클래스 처리
        with Pool() as pool:
            pool.starmap(self._process_class, [(lock, row, output_dir, num_images_per_class, csv_file) for _, row in df.iterrows()])

        # 시간 측정 종료 및 출력
        end_time = time.time()
        print(f"총 소요 시간: {end_time - start_time:.2f} 초")


    def _process_class(self, lock, row, output_dir, num_images_per_class, csv_file):
        class_name = row["class_name"]
        class_target = row["target"]
        class_text = row["text"]

        self.save_and_update_csv(lock, class_name, class_target, class_text, output_dir, num_images_per_class, csv_file)