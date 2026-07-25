# Filtering Alignment Tax: Semantic Steady-State Construction for Long-Range Reasoning in LLMs via Principal Component Deviation Suppression

**论文标题（中文）：** 《过滤对齐税：基于主元偏移抑制的大模型长程推理语义稳态构建》

**作者：** 孟强（独立研究者）

**通讯邮箱：** mengqiang.meigui@outlook.com

---

## 仓库内容说明

本仓库包含论文《过滤对齐税：基于主元偏移抑制的大模型长程推理语义稳态构建》的全部实验数据与核心脚本，供研究复现与同行评审使用。

### 数据文件夹说明

| 文件夹名称 | 对应实验条件 | 模型配置 | 题纲类型 |
|-----------|-------------|---------|---------|
| `原生苹果` | 0.5B 原生推理基线 | 0.5B 单模型 | 干净题纲 |
| `原生干扰苹果` | 0.5B 原生推理（干扰条件） | 0.5B 单模型 | 干扰题纲 |
| `双模型交替执行` | 0.5B 双模型协同框架 | 0.5B 生成 + 0.5B 执行 | 干净题纲 |
| `双模型交替_干扰` | 0.5B 双模型协同框架（干扰条件） | 0.5B 生成 + 0.5B 执行 | 干扰题纲 |
| `1.5B原生干净` | 1.5B 原生推理基线 | 1.5B 单模型 | 干净题纲 |
| `1.5B原生干扰` | 1.5B 原生推理（干扰条件） | 1.5B 单模型 | 干扰题纲 |
| `0.5B_理想样本_1.5B_点积` | 异构协同框架 | 0.5B 生成 + 1.5B 执行 | 干净题纲 |
| `0.5B_理想样本_1.5B_点积_干扰` | 异构协同框架（干扰条件） | 0.5B 生成 + 1.5B 执行 | 干扰题纲 |
| `消融一` | 移除参考信号，仅保留错误检测 | 0.5B 单模型 | 干净题纲 |
| `消融二` | 关闭回看机制，保留参考信号 | 0.5B 双模型 | 干净题纲 |

### 脚本文件说明

| 脚本名称 | 对应实验 |
|---------|---------|
| `run_dual_alternating.py` | 0.5B 双模型协同框架（干净/干扰） |
| `run_collaboration_0.5_to_1.5.py` | 异构协同框架（0.5B生成 + 1.5B执行） |
| `run_single_model_ablation.py` | 消融一：移除参考信号 |
| `run_dual_alternating_no_rollback.py` | 消融二：关闭回看机制 |

### 数据格式说明

每个数据文件夹包含：
- `logs.csv`：每次推理的详细日志（耗时、CPU占用、错误类型、对齐度等）
- `回答_XXX.txt`：模型原始输出文本

### 实验环境

- 模型：Qwen2.5-0.5B-Instruct、Qwen2.5-1.5B-Instruct
- 推理后端：CPU-only（AMD Ryzen 5 5600G）
- 生成策略：贪心确定性生成（temperature=0, top_p=1）
- Python 依赖：transformers、torch、psutil

### 复现说明

详细算法逻辑见论文附录B。完整实验脚本位于本仓库根目录，可直接运行。

### 许可证

MIT License

### 引用信息
@article{meng2025filtering,
title={过滤对齐税：基于主元偏移抑制的大模型长程推理语义稳态构建},
author={孟强},
journal={待投稿},
year={2025}
}

---

如有问题，请联系作者：mengqiang.meigui@outlook.com
