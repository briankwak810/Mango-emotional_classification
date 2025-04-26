from huggingface_hub import HfApi, upload_file
import os

HUGGINGFACE_USERNAME = "kjsbrian"
REPO_NAME = "mango-recall-classifier"
MODEL_FOLDER = "./mango-recall-classifier"

api = HfApi()

full_repo_name = f"{HUGGINGFACE_USERNAME}/{REPO_NAME}"

try:
    api.create_repo(repo_id=REPO_NAME, repo_type="model", exist_ok=True)
    print(f"Repo `{full_repo_name}` 준비 완료!")
except Exception as e:
    print(f"Repo 생성 중 에러 발생: {e}")

for filename in os.listdir(MODEL_FOLDER):
    file_path = os.path.join(MODEL_FOLDER, filename)
    if os.path.isfile(file_path):
        try:
            upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=full_repo_name,
                repo_type="model"
            )
            print(f"✅ {filename} 업로드 성공!")
        except Exception as e:
            print(f"업로드 중 에러 ({filename}): {e}")
