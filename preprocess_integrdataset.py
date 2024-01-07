import os
import shutil

src_path = r"D:\CV_Project\dataset"
dst_path = r"D:\CV_Project\dataset_integrated"

for root, dirs, files in os.walk(src_path):
    for file in files:
        if "GT" in file or "Parameters" in file:
            src_file_path = os.path.join(root, file)
            dst_file_path = os.path.join(dst_path, os.path.basename(root), file)
            os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
            shutil.copy(src_file_path, dst_file_path)
