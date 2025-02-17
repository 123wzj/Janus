import csv
import os

# 文件路径
SRC = 'D:/Python_File/Euler-main/data//auth.txt'
DEST = 'D:/Python_File/Janus/dataset/lanl/auth.csv'

# 定义列名
column_names = ['timestamp', 'src_user', 'dst_user', 'src_computer', 'dst_computer',
                'auth_type', 'logon_type', 'auth_orient', 'result']


def filter_and_save_auth_data_with_progress(src, dest):
    with open(src, 'r') as infile, open(dest, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # 写入列名
        writer.writerow(column_names)

        processed_lines = 0

        for row in reader:
            writer.writerow(row)

            processed_lines += 1
            # 每1000行显示一次处理进度
            if processed_lines % 1000 == 0:
                print(f"已处理 {processed_lines} 行", end='\r')

    print("\n处理完成！")


# 调用函数
filter_and_save_auth_data_with_progress(SRC, DEST)
