import re
from collections import defaultdict

def parse_cuda_timers(log_text):
    pattern = r'\[CUDA TIMER\]\s+([\w\s\+]+?)\s*:\s+([\d.]+)\s+ms'
    stages = defaultdict(list)
    for line in log_text.splitlines():
        match = re.search(pattern, line)
        if match:
            stage = match.group(1).strip()
            time_ms = float(match.group(2))
            stages[stage].append(time_ms)
    return stages

def compute_stats(stages):
    print(f"{'Stage':<35} {'Count':>8} {'Mean (ms)':>12} {'Std (ms)':>12} {'Percent (%)':>12}")
    print("-" * 85)
    # 先计算每个阶段的总耗时，用于比例
    stage_means = {}
    stage_counts = {}
    for stage, times in stages.items():
        mean = sum(times) / len(times)
        stage_means[stage] = mean
        stage_counts[stage] = len(times)
    total_mean = sum(stage_means.values())  # 平均每帧总时间（各阶段之和）
    for stage in sorted(stage_means.keys()):
        mean = stage_means[stage]
        count = stage_counts[stage]
        # 计算标准差
        times = stages[stage]
        variance = sum((t - mean) ** 2 for t in times) / len(times)
        std = variance ** 0.5
        percent = (mean / total_mean) * 100 if total_mean > 0 else 0
        print(f"{stage:<35} {count:>8} {mean:>12.4f} {std:>12.4f} {percent:>11.2f}%")

if __name__ == "__main__":
    # 请将日志粘贴到一个文本文件中，例如 log.txt，然后读取
    with open("log.txt", "r") as f:
        log = f.read()
    stages = parse_cuda_timers(log)
    compute_stats(stages)