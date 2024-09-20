from gan.gan_csv import GanCSV
from gan.gan_image_save import StableDiffusionImageGenerator

GanCSV()

# 인스턴스 생성
generator = StableDiffusionImageGenerator()

class_info_csv = "./gan/sorted_unique_class_name_class_text_data.csv"
output_dir = "./data/train/generated_images"
num_images_per_class = 1
csv_file = "./gan/updated_dataset_with_generated_images.csv"

# 모든 클래스에 대해 이미지 생성 및 CSV 업데이트 실행
generator.generate_images_for_classes(class_info_csv, output_dir, num_images_per_class, csv_file)
