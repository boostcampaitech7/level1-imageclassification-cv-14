import pandas as pd

# CSV 파일 읽기
csv_file_path = "/data/ephemeral/home/level1-imageclassification-cv-14/data/merged_train_with_generated.csv"  # 기존 CSV 파일 경로
df = pd.read_csv(csv_file_path)

# 경로 수정: generated_images 폴더를 생성된 이미지 경로 앞에 추가
df['image_path'] = df['image_path'].apply(lambda x: f"generated_images/{x}" if "generated" in x else x)

# 수정된 결과를 다시 CSV 파일에 저장
df.to_csv(csv_file_path, index=False)

print("경로가 수정된 CSV 파일이 저장되었습니다.")