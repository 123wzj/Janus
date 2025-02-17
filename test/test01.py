import os
import pickle

graphs_file = 'D:/Python_File/Janus/dataset/output/graphs_20250117_051246.pkl'
output_folder = 'D:/Python_File/Janus/dataset'

print(f"正在加载图数据从: {graphs_file}")
with open(graphs_file, 'rb') as f:
    graphs = pickle.load(f)


resume_from = None
if resume_from is None:
    resume_from = 0
    for idx in range(len(graphs)):
        model_path = f"{output_folder}/output/pth/short_term_snapshot_{idx}.pth"
        if os.path.exists(model_path):
            resume_from = idx + 1

print(f"从快照 {resume_from} 开始/继续训练")