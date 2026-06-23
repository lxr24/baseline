# 复现 StraightPCF 的步骤

本分支已经将 CVPR 2024 的论文 [StraightPCF](https://github.com/ddsediri/StraightPCF) 中的两个主要模型模块（`CoupledVMArch` 和 `StraightPCFArch`）完整移植至 Jittor 框架。我们为你规划了三阶段的多阶段接力训练流程。

### 第一阶段：训练 Velocity Module (VM)

首先运行基线的 VM 训练模块：
```bash
python run.py --task configs/task/train_vm.yaml
```
训练结束后，最佳权重将保存在 `experiments/` 目录中。

### 第二阶段：训练 Coupled VM (CVM)

修改 `configs/model/cvm.yaml`：
在其中加入：`velocity_ckpt: <上一步训练好的 vm_best_model.pkl>`

然后运行 CVM 训练：
```bash
python run.py --task configs/task/train_cvm.yaml
```

### 第三阶段：端到端训练 StraightPCF 全模型

修改 `configs/model/straightpcf.yaml`：
在其中加入：`cvm_ckpt: <上一步训练好的 cvm_best_model.pkl>`

然后运行 StraightPCF 的最终训练：
```bash
python run.py --task configs/task/train_straightpcf.yaml
```

### 推理测试

模型训练完毕后，你可以分别进行测试并导出降噪点云：
```bash
python run.py --task configs/task/predict_cvm.yaml
# 或者
python run.py --task configs/task/predict_straightpcf.yaml
```
确保在 `predict` 配置文件中配置正确的 `load_ckpt`。
