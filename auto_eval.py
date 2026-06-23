import os
import subprocess
import yaml # 如果没有安装，可以通过 pip install pyyaml 安装

def update_yaml_ckpt(yaml_path, new_ckpt_path):
    """安全地更新 predict_vm.yaml 中的 load_ckpt 路径"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        for line in lines:
            if line.strip().startswith('load_ckpt:'):
                f.write(f'load_ckpt: {new_ckpt_path}\n')
            else:
                f.write(line)

def run_evaluation():
    # ================= 配置区 =================
    # 你想测试的 Epoch 列表 (从 60 到 100，每 5 轮测一次)
    epochs_to_test = list(range(60, 101, 5)) 
    
    # 文件路径配置
    exp_dir = "experiments/vm"
    predict_yaml_path = "/home/lixinran/baseline/configs/task/predict_vm.yaml"
    
    # 请根据你的实际路径修改以下两个目录
    pred_dir = "./results/dataset_train"  # 你的模型预测保存点云的目录
    gt_dir = "./dataset_train"              
    mesh_dir = "./dataset_train"
    # ==========================================

    best_epoch = -1
    best_score = float('-inf')
    results = {}
    
    print("🚀 开始执行自动化本地打榜流程...")
    
    for epoch in epochs_to_test:
        ckpt_path = os.path.join(exp_dir, f"checkpoint_{epoch}.pkl")
        if not os.path.exists(ckpt_path):
            print(f"⚠️ 找不到权重文件: {ckpt_path}，跳过。")
            continue
            
        print(f"\n" + "="*50)
        print(f"🎯 正在测试 Epoch: {epoch}")
        
        # 1. 自动修改 yaml 中的 checkpoint 路径
        update_yaml_ckpt(predict_yaml_path, ckpt_path)
        
        # 2. 运行推理生成点云
        print(">> 正在运行模型推理...")
        run_cmd = "python run.py --task configs/task/predict_vm.yaml"
        result_code = os.system(run_cmd)
        if result_code != 0:
            print(f"❌ Epoch {epoch} 推理出错，跳过。")
            continue
            
        # 3. 运行官方评测脚本计算 CD/P2S
        print(">> 正在计算几何评价指标...")
        eval_cmd = f"python evaluate.py --pred_dir {pred_dir} --gt_dir {gt_dir} --noisy_dir {gt_dir} --mesh_dir {mesh_dir} --workers 8"
        
        try:
            # 捕获 evaluate.py 的终端输出
            output = subprocess.check_output(eval_cmd, shell=True, text=True)
            print(output.strip())
            
            # 解析最终得分 (假设你的 evaluate.py 最后会输出类似 "Final Score: 85.5")
            # ⚠️ 注意：这里需要根据你 evaluate.py 实际打印的格式来调整切词逻辑！
            score = 0.0
            for line in output.split('\n'):
                # 寻找包含核心指标的行，提取数字
                if "Score" in line or "Overall" in line or "CD:" in line: 
                    # 这里粗略提取最后一个浮点数作为成绩
                    parts = line.split()
                    for p in reversed(parts):
                        try:
                            score = float(p.strip())
                            break
                        except ValueError:
                            continue
            
            results[epoch] = score
            if score > best_score:
                best_score = score
                best_epoch = epoch
                print(f"🌟 发现当前最高分！Epoch {epoch} 得分: {best_score}")
                
        except Exception as e:
            print(f"❌ 评测 Epoch {epoch} 时出错: {e}")

    # 输出最终排行榜
    print("\n" + "="*50)
    print("🏆 粗筛完成！所有测试的 Epoch 成绩单：")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for ep, sc in sorted_results:
        print(f"   - Epoch {ep}: {sc}")
        
    print(f"\n🥇 初步最好的一轮是第 {best_epoch} 轮，得分: {best_score}")
    print(">> 接下来，请挑选排名前三的 Epoch，将 seed_k 拉满进行最后冲刺评估！")

if __name__ == "__main__":
    run_evaluation()