import pandas as pd

class GanCSV:
    def __init__(self):
        self.create_and_save_csv()  # 클래스가 생성될 때 바로 메서드 실행

    def create_and_save_csv(self):
        # 두 개의 CSV 파일을 읽기
        class_to_text_df = pd.read_csv("./gan/class_to_text.csv")  # class와 text 매핑이 있는 파일
        image_data_df = pd.read_csv("./data/train.csv")  # class_name, image_path, target이 있는 파일

        # 1. class_to_text_df에서 "class"와 "text" 컬럼을 사용하여 병합
        merged_df = pd.merge(image_data_df[['class_name', 'target']], class_to_text_df[['class', 'text']], how="left", left_on="target", right_on="class")

        # 2. 각 class에 대해 고유한 class_name만 남기기 (중복 제거)
        final_df = merged_df.drop_duplicates(subset='target')

        # 3. 필요한 열만 선택 (class_name, target, text)
        final_df = final_df[['class_name', 'target', 'text']]

        # 4. target 숫자 오름차순으로 정렬
        final_df = final_df.sort_values(by='target')

        # 5. 결과 CSV로 저장
        final_df.to_csv("./gan/sorted_unique_class_name_class_text_data.csv", index=False)

        print(f"정렬된 데이터를 'sorted_unique_class_name_class_text_data.csv'로 저장했습니다. 총 {len(final_df)}개의 행이 있습니다.")