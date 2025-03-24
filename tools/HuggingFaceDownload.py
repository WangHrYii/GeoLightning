from huggingface_hub import snapshot_download
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 下载仓库的所有文件
repo_id = "IGNF/TreeSatAI-Time-Series"  # 替换为你想下载的模型仓库 ID
local_dir = "/mnt/data_1/IGNF/TreeSatAI-Time-Series"       # 本地存储路径
snapshot_download(repo_id=repo_id, 
                  repo_type='dataset',
                  local_dir=local_dir,
                  resume_download=True)

print(f"Repository downloaded to: {local_dir}")
