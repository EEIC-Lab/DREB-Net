import os
import argparse

def get_file_list(folder_path, is_abspath=True, extensions=['jpg', 'png']):
    # folder_path - 需要遍历的文件夹路径
    # is_abspath - 是否返回遍历文件的绝对路径
    # extensions - 需要遍历文件的类型
    # return - 返回folder_path下需要遍历文件的路径list
    file_list = []
    file_name = None
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for file in filenames:
            ext = file.split('.')[-1]
            file_name = file.split('.')[0]
            if ext in extensions:
                if is_abspath:
                    file_list.append(os.path.join(dirpath, file))
                else:
                    file_list.append(file)

    return sorted(file_list)

