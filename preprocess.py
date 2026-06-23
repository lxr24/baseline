import os
import glob
import numpy as np
import trimesh
from tqdm import tqdm

# 确保能正确引入你的采样函数
from src.data.utils import sample_vertex_groups

def preprocess_obj_to_npy(dataset_dir, num_samples=100000):
    """
    将 obj 网格高密度采样为紧凑的 .npy 文件，消除训练时的解析与重心采样开销
    """
    obj_files = glob.glob(os.path.join(dataset_dir, "**", "*.obj"), recursive=True)
    print(f"🚀 找到 {len(obj_files)} 个 .obj 文件，开始离线采样...")
    
    for obj_path in tqdm(obj_files):
        npy_path = obj_path.replace('.obj', '.npy')
        if os.path.exists(npy_path):
            continue  # 如果已经生成过，直接跳过
        
        try:
            # 读取网格
            mesh = trimesh.load(obj_path, process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
                
            # 抽取高密度的表面点（10万点，足以完美逼近真实表面）
            sampled_vertices, _, _, _ = sample_vertex_groups(
                vertices=np.array(mesh.vertices),
                faces=np.array(mesh.faces),
                num_samples=num_samples
            )
            
            # 存为 float32 的二进制格式，极大幅度减小 I/O 体积
            np.save(npy_path, sampled_vertices.astype(np.float32))
        except Exception as e:
            print(f"❌ 处理 {obj_path} 失败: {e}")

if __name__ == "__main__":
    # 替换为你实际存放数据集的根目录
    DATA_ROOT = "/home/lixinran/baseline/dataset_train"
    preprocess_obj_to_npy(DATA_ROOT)