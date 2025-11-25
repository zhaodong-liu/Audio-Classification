import os
import shutil

def move_files_by_index(txt_file, source_dir):
    
    # read txt file
    with open(txt_file, 'r') as file:
        cnt = 0
        for line in file:
            # if cnt != 11228:
            #     cnt += 1
            #     continue
            label = line.strip()
            # print(label)
            # break
            filename = str(cnt) + '.mp3'
            source_path = os.path.join(source_dir, filename)
            print(source_path)
            
            # if the file exists, move it to the target directory
            if os.path.exists(source_path):
                target_path = label + '/' + filename
                shutil.move(source_path, target_path)
                print(f"{filename} moved to {target_path}")
            else:
                print(f"{filename} not found in the source directory.")
            cnt += 1
            

# 使用示例
txt_file = 'train_label.txt'  # label text file
source_dir = "train_mp3s/"  # source directory

move_files_by_index(txt_file, source_dir)
