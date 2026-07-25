# run_dual_alternating_no_rollback.py
# 双模型交替执行 - 回看消融实验（20次）
# 关闭回看机制，保持检测规则启用
# 模2提供参考信号，模1执行推理，不触发回退重试
# 用于验证回看机制在框架中的独立贡献

import os
import sys
import time
import csv
import psutil
import torch
import re
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_PATH, DATA_DIR, LOG_HEADER
from model_loader import load_model
from algo_hook_v1_2 import AlgorithmHookV12


PROCESS = psutil.Process()

# 两个0.5实例的路径（与当前90%正确率版本完全一致）
MODEL_A_PATH = r"C:\Users\MSN\Qwen2.5-0.5B-Instruct"  # 模1（尾元，执行推理）
MODEL_B_PATH = r"C:\Users\MSN\Qwen2.5-0.5B-Instruct_Original"  # 模2（主元，提供参考信号）

# 干净题目
CLEAN_PROMPT = (
    "Xiao Ming has 3 apples. "
    "Xiao Hong has 2 more apples than Xiao Ming. "
    "Xiao Hua has half as many apples as Xiao Hong. "
    "Xiao Hua gives 1 apple to Xiao Ming. "
    "How many apples does Xiao Ming have now? "
    "Please write out your reasoning step by step."
)

# 理想样本（与当前90%正确率版本完全一致）
IDEAL_SAMPLE = (
    "Let's break down the problem step by step:\n\n"
    "1. Initial number of apples:\n"
    "   - Xiao Ming has 3 apples.\n"
    "   - Xiao Hong has 2 more apples than Xiao Ming.\n"
    "     Number of apples Xiao Hong has = 3 + 2 = 5\n\n"
    "2. Number of apples Xiao Hua has:\n"
    "   - Xiao Hua has half as many apples as Xiao Hong.\n"
    "     Number of apples Xiao Hua has = 5 / 2 = 2.5\n\n"
    "3. Apples given to Xiao Ming:\n"
    "   - Xiao Hua gives 1 apple to Xiao Ming.\n"
    "   - Xiao Ming's new total = 3 + 1 = 4\n\n"
    "Therefore, after giving 1 apple to Xiao Ming, Xiao Ming has 4 apples."
)

# ========== 消融实验配置 ==========
CONFIG = {
    "alignment_threshold": 0.7,
    "max_rounds": 3,
    "enable_rollback": False,  # 关闭回看
}
# =================================

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def sample_cpu():
    return PROCESS.cpu_percent(interval=0.05)

def sample_memory():
    return PROCESS.memory_info().rss / (1024 * 1024)


def load_model_a():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f">>> 加载模1（尾元）: {MODEL_A_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_A_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_A_PATH,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    ).eval()
    return tokenizer, model


def load_model_b():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f">>> 加载模2（主元）: {MODEL_B_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_B_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_B_PATH,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    ).eval()
    return tokenizer, model


def generate_response(model, tokenizer, prompt):
    """执行单次推理，返回响应文本"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=512)
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    return response


def run_model2_with_ideal(model2, tokenizer2, prompt):
    """模2：以理想样本为参考，生成推理路径"""
    enhanced_prompt = (
        f"{prompt}\n\n"
        f"Reference solution:\n{IDEAL_SAMPLE}\n\n"
        f"Based on the reference solution, solve the problem step by step."
    )
    return generate_response(model2, tokenizer2, enhanced_prompt)


def compute_alignment(response1: str, response2: str, model, tokenizer) -> float:
    """计算模1和模2输出的对齐程度（点积距离）"""
    with torch.no_grad():
        tokens1 = tokenizer(response1, return_tensors="pt")["input_ids"].to(model.device)
        embeds1 = model.get_input_embeddings()(tokens1)
        avg1 = embeds1.mean(dim=1).squeeze(0)
        
        tokens2 = tokenizer(response2, return_tensors="pt")["input_ids"].to(model.device)
        embeds2 = model.get_input_embeddings()(tokens2)
        avg2 = embeds2.mean(dim=1).squeeze(0)
        
        cos_sim = torch.nn.functional.cosine_similarity(avg1.unsqueeze(0), avg2.unsqueeze(0)).item()
        
    return cos_sim


def detect_error(response: str) -> tuple:
    """错误检测规则（与当前90%正确率版本完全一致）"""
    has_error = False
    error_types = []
    severity = "none"

    if "4" in response:
        if "Xiao Hua" not in response or "half" not in response:
            error_types.append("静默跳跃-无害型")
            severity = "low"
            has_error = True
    else:
        has_error = True
        if "Xiao Ming gives" in response:
            error_types.append("语义角色漂移")
            severity = "high"
        elif "minus" in response or "3 - 1" in response:
            error_types.append("运算方向反转")
            severity = "high"
        elif "Xiao Hua has 3" in response or "Xiao Hua has 2.5" in response:
            error_types.append("题设覆盖")
            severity = "medium"
        elif "total" in response.lower() or "sum" in response.lower():
            error_types.append("局部自洽全局崩溃")
            severity = "high"
        else:
            error_types.append("目标漂移")
            severity = "high"
    
    return has_error, error_types, severity


def run_dual_alternating_no_rollback(model1, tokenizer1, model2, tokenizer2, prompt, index):
    """双模型交替执行 - 回看关闭版本"""
    cpu_before = sample_cpu()
    start_time = time.time()
    start_time_str = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")

    # 初始化算法三
    algo = AlgorithmHookV12()
    
    model2_responses = []
    model1_responses = []
    alignments = []
    
    current_prompt = prompt
    round_num = 0
    max_rounds = CONFIG["max_rounds"]
    enable_rollback = CONFIG["enable_rollback"]
    
    while round_num < max_rounds:
        round_num += 1
        
        # 步骤1：模2先推理（以理想样本为参考）
        response2 = run_model2_with_ideal(model2, tokenizer2, current_prompt)
        model2_responses.append(response2)
        
        # 步骤2：模1对着模2的结果推理
        prompt1 = (
            f"{prompt}\n\n"
            f"Reference reasoning path:\n{response2}\n\n"
            f"Based on the reference path, solve the problem step by step."
        )
        response1 = generate_response(model1, tokenizer1, prompt1)
        model1_responses.append(response1)
        
        # 步骤3：计算对齐程度
        alignment = compute_alignment(response1, response2, model1, tokenizer1)
        alignments.append(alignment)
        
        # ========== 回看消融：关闭回看 ==========
        # 当 enable_rollback = False 时，跳过回看触发逻辑
        # 模1只执行一次推理，不进行重试
        if enable_rollback:
            # 如果对齐度低于阈值，触发回看（仅在启用时执行）
            if alignment < CONFIG["alignment_threshold"]:
                retry_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous reasoning had low alignment with the reference. "
                    f"Please try again. Reference path:\n{response2}\n\n"
                    f"Solve the problem step by step."
                )
                response1_retry = generate_response(model1, tokenizer1, retry_prompt)
                response1 = response1_retry
        
        # 检查是否已得到最终答案
        if "4" in response1 or "answer is 4" in response1:
            break
        
        # 更新当前输入
        current_prompt = (
            f"{prompt}\n\n"
            f"Previous reasoning:\n{response1}\n\n"
            f"Continue the reasoning."
        )
    
    final_response = response1 if response1 else ""
    
    # 错误检测
    has_error, error_types, severity = detect_error(final_response)
    
    end_time = time.time()
    end_time_str = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
    duration = end_time - start_time
    memory_used = sample_memory()

    log_entry = {
        "timestamp": get_timestamp(),
        "stage": "双模型交替_回看关闭",
        "question_index": index,
        "question_text": prompt[:80] + "..." if len(prompt) > 80 else prompt,
        "start_time": start_time_str,
        "end_time": end_time_str,
        "duration_seconds": round(duration, 4),
        "cpu_before": round(cpu_before, 2),
        "memory_used_mb": round(memory_used, 2),
        "has_error": has_error,
        "error_types": ",".join(error_types),
        "severity": severity,
        "rounds": round_num,
        "alignments": str([round(a, 3) for a in alignments]),
        "rollback_enabled": enable_rollback,
        "final_response": final_response[:200] + "..." if len(final_response) > 200 else final_response,
        "answer": "4" if "4" in final_response else "wrong",
    }

    return final_response, log_entry


def main():
    print("\n" + "=" * 50)
    print("    双模型交替执行 - 回看消融实验（20次）")
    print("    回看机制：关闭")
    print("    检测规则：启用")
    print("=" * 50)

    # 加载两个模型
    tokenizer1, model1 = load_model_a()
    tokenizer2, model2 = load_model_b()

    stage_name = "双模型交替_回看关闭"
    stage_folder = os.path.join(DATA_DIR, stage_name)
    os.makedirs(stage_folder, exist_ok=True)

    log_path = os.path.join(stage_folder, "logs.csv")
    log_exists = os.path.isfile(log_path)

    extended_header = LOG_HEADER + [
        "has_error", "error_types", "severity",
        "rounds", "alignments", "rollback_enabled",
        "final_response", "answer"
    ]

    print(f"\n>>> 开始运行，共 20 次推理")
    print(f"    存储目录: {stage_name}\n")

    for i in range(1, 21):
        print(f"  [{i}/20] 推理中...", end=" ", flush=True)

        response, log_entry = run_dual_alternating_no_rollback(
            model1, tokenizer1,
            model2, tokenizer2,
            CLEAN_PROMPT, i
        )

        txt_path = os.path.join(stage_folder, f"回答_{i:03d}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(response)

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=extended_header)
            if not log_exists:
                writer.writeheader()
                log_exists = True
            writer.writerow(log_entry)

        print(f"完成。耗时 {log_entry['duration_seconds']:.2f}s | 错误: {log_entry['has_error']}")

    print(f"\n>>> 双模型交替执行 - 回看消融实验完成！")
    print(f"    数据保存在: {stage_folder}")
    print("=" * 50)


if __name__ == "__main__":
    main()