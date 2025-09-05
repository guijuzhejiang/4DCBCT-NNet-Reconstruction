import os
import shutil

# 根目录路径
root_dir = "/home/zzg/data/Medical/4D_Lung_CBCT_Hitachi/dataset"

# 三个主目录
main_dirs = ["FovL", "FovS_180", "FovS_360"]

# 要删除的子目录名称
dirs_to_remove = ["phase_01", "phase_02", "phase_03", "phase_04"]

# 遍历每个主目录
for main_dir in main_dirs:
    main_path = os.path.join(root_dir, main_dir)

    # 遍历每个subject目录 (0000-0049)
    for subject_id in range(50):
        subject_dir = f"subject_{subject_id:04d}"  # 格式化为4位数字
        subject_path = os.path.join(main_path, subject_dir)

        # 检查subject目录是否存在
        if not os.path.exists(subject_path):
            print(f"警告: 目录不存在 {subject_path}")
            continue

        # 遍历需要删除的目录
        for dir_name in dirs_to_remove:
            target_dir = os.path.join(subject_path, dir_name)

            # 检查目标目录是否存在
            if os.path.exists(target_dir):
                try:
                    # 递归删除目录
                    shutil.rmtree(target_dir)
                    print(f"已删除: {target_dir}")
                except Exception as e:
                    print(f"删除失败 {target_dir}: {str(e)}")
            else:
                print(f"目录不存在，跳过: {target_dir}")

print("操作完成")