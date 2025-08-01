from huggingface_hub import snapshot_download

# 下载整个数据集
snapshot_download(repo_id="initiacms/OpticalRS-4M", repo_type="dataset", local_dir="/mnt/data/OpticalRS-4M", force_download=True, resume_download=True)
