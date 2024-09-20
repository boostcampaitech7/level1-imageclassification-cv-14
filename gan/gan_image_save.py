import os
import pandas as pd
from diffusers import StableDiffusionPipeline
from PIL import Image
from multiprocessing import Pool, cpu_count

class StableDiffusionImageGenerator:
    def __init__(self, model_name: str = "CompVis/stable-diffusion-v1-4", device: str = "cuda"):
        # Stable Diffusion 모델 로드
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name).to(device)

    # 스케치 이미지를 생성하는 메서드
    def generate_sketch(self, prompt: str, num_images: int = 1):
        images = self.pipe(prompt, num_inference_steps=50).images  # 이미지 생성
        return images

    # 이미지를 저장하고 CSV 파일에 추가하는 메서드
    def save_and_update_csv(self, class_name, class_target, text, output_dir, num_images, csv_file):
        # 이미지 저장 경로 생성
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # CSV 파일 읽기
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["class_name", "image_path", "target"])

        # 이미지 생성 및 저장
        for i in range(num_images):
            # 이미지 파일명 생성
            image_file = f"{class_name}_generated_{i}.JPEG"
            image_path = os.path.join(class_dir, image_file)

            # 이미지가 이미 존재하는지 확인 (중복 방지)
            if not os.path.exists(image_path):
                # 프롬프트에 기반한 스케치 생성
                prompt = f"a sketch of {text}"  # 클래스 이름에 맞는 프롬프트
                images = self.generate_sketch(prompt, num_images=1)

                # 이미지 저장
                images[0].save(image_path)

                # CSV에 중복된 이미지 정보가 있는지 확인 후 추가
                if not ((df['class_name'] == class_name) & (df['image_path'] == image_path)).any():
                    # 새로운 행을 DataFrame으로 생성
                    new_row = pd.DataFrame({"class_name": [class_name], "image_path": [image_path], "target": [class_target]})
                    # concat을 사용하여 기존 DataFrame에 새로운 행 추가
                    df = pd.concat([df, new_row], ignore_index=True)
        
        # CSV 파일 저장
        df.to_csv(csv_file, index=False)

    # 멀티프로세싱을 위한 작업 단위
    def generate_images_for_class(self, row, output_dir, num_images_per_class, csv_file):
        class_name = row["class_name"]
        class_target = row["target"]
        class_text = row["text"]
        
        self.save_and_update_csv(class_name, class_target, class_text, output_dir, num_images_per_class, csv_file)

    # 멀티 프로세싱을 통한 이미지 생성 및 CSV 업데이트
    def generate_images_for_classes(self, class_info_csv, output_dir, num_images_per_class, csv_file):
        # 클래스 정보가 담긴 CSV 파일을 읽음
        df = pd.read_csv(class_info_csv)

        # 멀티프로세싱을 사용하여 각 클래스에 대해 병렬적으로 이미지 생성
        with Pool(cpu_count()) as pool:
            pool.starmap(self.generate_images_for_class, [(row, output_dir, num_images_per_class, csv_file) for _, row in df.iterrows()])