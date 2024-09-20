from gan.gan_csv import GanCSV
from gan.gan_image_save import StableDiffusionImageGenerator
from multiprocessing import set_start_method
import pandas as pd

# 메인 실행 부분을 __main__ 블록 안에 넣어줍니다.
if __name__ == "__main__":
    # 멀티프로세싱을 위해 'spawn' 시작 방식을 설정
    set_start_method('spawn', force=True)

    # 1. 먼저 GanCSV()를 실행하여 CSV 파일 생성
    GanCSV()

    # 2. 인스턴스 생성
    generator = StableDiffusionImageGenerator()

    # 3. 클래스 정보가 담긴 CSV 파일 경로
    class_info_csv = "./gan/sorted_unique_class_name_class_text_data.csv"
    
    # 4. 생성된 이미지를 저장할 디렉토리
    output_dir = "./data/train/generated_images"
    
    # 5. 클래스당 생성할 이미지 수
    num_images_per_class = 5
    
    # 6. 업데이트된 CSV 파일 경로
    csv_file = "./gan/updated_dataset_with_generated_images.csv"

    # 7. 모든 클래스에 대해 이미지 생성 및 CSV 업데이트 실행
    generator.generate_images_for_classes(class_info_csv, output_dir, num_images_per_class, csv_file)

    # 8. 기존의 train.csv 파일과 생성된 CSV 파일을 병합하는 부분
    train_csv_path = "./data/train.csv"  # 기존 train.csv 파일 경로
    merged_csv_output_path = "./data/merged_train_with_generated.csv"  # 병합된 파일 저장 경로

    # 9. 기존의 train.csv 파일 읽기
    train_df = pd.read_csv(train_csv_path)

    # 10. 새로 생성된 CSV 파일 읽기
    generated_df = pd.read_csv(csv_file)

    # 11. 두 데이터프레임 결합 (중복 방지를 위해 인덱스를 새롭게 할당)
    merged_df = pd.concat([train_df, generated_df], ignore_index=True)

    # 12. 병합된 CSV 파일을 저장
    merged_df.to_csv(merged_csv_output_path, index=False)

    print(f"CSV 파일이 성공적으로 병합되었습니다: {merged_csv_output_path}")