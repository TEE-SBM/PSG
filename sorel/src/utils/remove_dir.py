import os
import shutil

def delete_prg_folders(root_dir, flag):
    # 遍历第一级目录
    for first_level in os.listdir(root_dir):
        first_level_path = os.path.join(root_dir, first_level)
        if not os.path.isdir(first_level_path):
            continue  # 跳过非目录项
        
        # 遍历第二级目录
        for second_level in os.listdir(first_level_path):
            second_level_path = os.path.join(first_level_path, second_level)
            if not os.path.isdir(second_level_path):
                continue  # 跳过非目录项
            
            # 遍历第三级目录
            for third_level in os.listdir(second_level_path):
                third_level_path = os.path.join(second_level_path, third_level)
                if os.path.isdir(third_level_path) and flag in third_level.lower():
                    try:
                        shutil.rmtree(third_level_path)
                        print(f"Deleted: {third_level_path}")
                    except Exception as e:
                        print(f"Failed to delete {third_level_path}: {e}")

# 示例：删除路径 "/path/to/root" 下的所有符合条件的文件夹
root_directory = "./results"  # 请修改为实际目录
flag = "psgsrmgrad"  # 请修改为实际的关键词
delete_prg_folders(root_directory, flag)
