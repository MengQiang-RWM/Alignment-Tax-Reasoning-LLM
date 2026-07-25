# run_single_model_ablation.py
# 单模型独立推理消融实验（20次）
# 模1独立推理，不包含回看，不包含模2参考信号
# 只应用错误检测规则记录正确率
# 用于确认双模型交替框架的正确率来源

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


PROCESS = psutil.Process()

# 干净题目
CLEAN_PROMPT = (
    "Xiao Ming has 3 apples. "
    "Xiao Hong has 2 more apples than Xiao Ming. "
    "Xiao Hua has half as many apples as Xiao Hong. "
    "Xiao Hua gives 1 apple to Xiao Ming. "
    "How many apples does Xiao Ming have now? "
    "Please write out your reasoning step by step."
)

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def sample_cpu():
    return PROCESS.cpu_percent(interval=0.05)

def sample_memory():
    return PROCESS.memory_info().rss / (1024 * 1024)


def generate_response(model, tokenizer, prompt):
    """执行单次推理，返回响应文本"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=512)
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    return response


def detect_error(response: str) -> tuple:
    """错误检测规则（与双模型交替框架完全一致）"""
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


def run_single_model_inference(model, tokenizer, prompt, index):
    """单模型独立推理"""
    cpu_before = sample_cpu()
    start_time = time.time()
    start_time_str = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")

    # 模1独立推理
    response = generate_response(model, tokenizer, prompt)

    end_time = time.time()
    end_time_str = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
    duration = end_time - start_time
    memory_used = sample_memory()

    # 应用错误检测规则
    has_error, error_types, severity = detect_error(response)

    log_entry = {
        "timestamp": get_timestamp(),
        "stage": "单模型消融",
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
        "final_response": response[:200] + "..." if len(response) > 200 else response,
        "answer": "4" if "4" in response else "wrong",
    }

    return response, log_entry


def main():
    print("\n" + "=" * 50)
    print("    单模型独立推理消融实验（20次）")
    print("    不包含回看，不包含模2参考信号")
    print("    只应用错误检测规则")
    print("=" * 50)

    tokenizer, model = load_model()

    stage_name = "单模型消融"
    stage_folder = os.path.join(DATA_DIR, stage_name)
    os.makedirs(stage_folder, exist_ok=True)

    log_path = os.path.join(stage_folder, "logs.csv")
    log_exists = os.path.isfile(log_path)

    extended_header = LOG_HEADER + [
        "has_error", "error_types", "severity",
        "final_response", "answer"
    ]

    print(f"\n>>> 开始运行，共 20 次推理")
    print(f"    存储目录: {stage_name}\n")

    for i in range(1, 21):
        print(f"  [{i}/20] 推理中...", end=" ", flush=True)

        response, log_entry = run_single_model_inference(
            model, tokenizer, CLEAN_PROMPT, i
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

    print(f"\n>>> 单模型独立推理消融实验完成！")
    print(f"    数据保存在: {stage_folder}")
    print("=" * 50)


if __name__ == "__main__":
    main()