# run_collaboration_0.5_to_1.5.py
# 0.5B（理想样本）→ 1.5B（点积）协同测试（20次）
# 0.5B生成理想样本作为参考信号
# 1.5B执行推理并进行点积对齐
# 复用0.5B上验证过的双模型交替框架

import os
import sys
import time
import csv
import psutil
import torch
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, LOG_HEADER


# 模型路径
MODEL_0_5B_PATH = r"C:\Users\MSN\Qwen2.5-0.5B-Instruct"  # 0.5B（理想样本生成）
MODEL_1_5B_PATH = r"C:\Users\MSN\qwen_model"  # 1.5B（执行推理）

# 干净题目
CLEAN_PROMPT = (
    "Xiao Ming has 3 apples. "
    "Xiao Hong has 2 more apples than Xiao Ming. "
    "Xiao Hua has half as many apples as Xiao Hong. "
    "Xiao Hua gives 1 apple to Xiao Ming. "
    "How many apples does Xiao Ming have now? "
    "Please write out your reasoning step by step."
)

# 理想样本（算法三最佳输出）
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

CONFIG = {
    "alignment_threshold": 0.7,
    "max_rounds": 3,
    "enable_rollback": True,
}

PROCESS = psutil.Process()

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def sample_cpu():
    return PROCESS.cpu_percent(interval=0.05)

def sample_memory():
    return PROCESS.memory_info().rss / (1024 * 1024)


def load_model_0_5b():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f">>> 加载0.5B模型（理想样本生成）: {MODEL_0_5B_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_0_5B_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_0_5B_PATH,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    ).eval()
    return tokenizer, model


def load_model_1_5b():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f">>> 加载1.5B模型（执行推理）: {MODEL_1_5B_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_1_5B_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_1_5B_PATH,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    ).eval()
    return tokenizer, model


def generate_response(model, tokenizer, prompt):
    """执行单次推理"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=512)
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    return response


def run_model_0_5b_ideal(model_0_5b, tokenizer_0_5b, prompt):
    """0.5B以理想样本为参考生成推理路径"""
    enhanced_prompt = (
        f"{prompt}\n\n"
        f"Reference solution:\n{IDEAL_SAMPLE}\n\n"
        f"Based on the reference solution, solve the problem step by step."
    )
    return generate_response(model_0_5b, tokenizer_0_5b, enhanced_prompt)


def compute_alignment(response1: str, response2: str, model, tokenizer) -> float:
    """计算对齐程度（点积距离）"""
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
    """错误检测规则"""
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


def run_collaboration(model_0_5b, tokenizer_0_5b, model_1_5b, tokenizer_1_5b, prompt, index):
    """0.5B（理想样本）→ 1.5B（点积）协同推理"""
    cpu_before = sample_cpu()
    start_time = time.time()
    start_time_str = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")

    model_1_5b_responses = []
    alignments = []
    corrections = []
    
    current_prompt = prompt
    round_num = 0
    max_rounds = CONFIG["max_rounds"]
    enable_rollback = CONFIG["enable_rollback"]
    
    while round_num < max_rounds:
        round_num += 1
        
        # 0.5B生成理想参考信号
        reference_response = run_model_0_5b_ideal(model_0_5b, tokenizer_0_5b, current_prompt)
        
        # 1.5B对着参考信号执行推理
        prompt_1_5b = (
            f"{prompt}\n\n"
            f"Reference reasoning path:\n{reference_response}\n\n"
            f"Based on the reference path, solve the problem step by step."
        )
        response_1_5b = generate_response(model_1_5b, tokenizer_1_5b, prompt_1_5b)
        model_1_5b_responses.append(response_1_5b)
        
        # 计算对齐程度
        alignment = compute_alignment(response_1_5b, reference_response, model_1_5b, tokenizer_1_5b)
        alignments.append(alignment)
        
        if enable_rollback:
            if alignment < CONFIG["alignment_threshold"]:
                retry_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous reasoning had low alignment with the reference. "
                    f"Please try again. Reference path:\n{reference_response}\n\n"
                    f"Solve the problem step by step."
                )
                response_1_5b_retry = generate_response(model_1_5b, tokenizer_1_5b, retry_prompt)
                corrections.append({
                    "round": round_num,
                    "alignment": alignment,
                    "original_response": response_1_5b,
                    "retry_response": response_1_5b_retry
                })
                response_1_5b = response_1_5b_retry
        
        if "4" in response_1_5b or "answer is 4" in response_1_5b:
            break
        
        current_prompt = (
            f"{prompt}\n\n"
            f"Previous reasoning:\n{response_1_5b}\n\n"
            f"Continue the reasoning."
        )
    
    final_response = response_1_5b if response_1_5b else ""
    
    has_error, error_types, severity = detect_error(final_response)
    
    end_time = time.time()
    end_time_str = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
    duration = end_time - start_time
    memory_used = sample_memory()

    log_entry = {
        "timestamp": get_timestamp(),
        "stage": "0.5B_理想样本_1.5B_点积",
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
        "corrections": len(corrections),
        "final_response": final_response[:200] + "..." if len(final_response) > 200 else final_response,
        "answer": "4" if "4" in final_response else "wrong",
    }

    return final_response, log_entry


def main():
    print("\n" + "=" * 50)
    print("    0.5B（理想样本）→ 1.5B（点积）协同测试")
    print("    0.5B生成理想参考信号")
    print("    1.5B执行推理 + 点积对齐")
    print("=" * 50)

    tokenizer_0_5b, model_0_5b = load_model_0_5b()
    tokenizer_1_5b, model_1_5b = load_model_1_5b()

    stage_name = "0.5B_理想样本_1.5B_点积"
    stage_folder = os.path.join(DATA_DIR, stage_name)
    os.makedirs(stage_folder, exist_ok=True)

    log_path = os.path.join(stage_folder, "logs.csv")
    log_exists = os.path.isfile(log_path)

    extended_header = LOG_HEADER + [
        "has_error", "error_types", "severity",
        "rounds", "alignments", "corrections",
        "final_response", "answer"
    ]

    print(f"\n>>> 开始运行，共 20 次推理")
    print(f"    存储目录: {stage_name}\n")

    for i in range(1, 21):
        print(f"  [{i}/20] 推理中...", end=" ", flush=True)

        response, log_entry = run_collaboration(
            model_0_5b, tokenizer_0_5b,
            model_1_5b, tokenizer_1_5b,
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

        print(f"完成。耗时 {log_entry['duration_seconds']:.2f}s | 错误: {log_entry['has_error']} | 轮数: {log_entry['rounds']}")

    print(f"\n>>> 0.5B（理想样本）→ 1.5B（点积）协同测试完成！")
    print(f"    数据保存在: {stage_folder}")
    print("=" * 50)


if __name__ == "__main__":
    main()