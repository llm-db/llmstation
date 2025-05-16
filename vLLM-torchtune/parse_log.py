import argparse
from pathlib import Path
import re


def fetch_line(filepath, keyword):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if keyword in line:
                return line


def calculate_throughput(filepath, start_line, end_line):
    pattern = re.compile(r"tokens_per_second_per_gpu:([\d\.]+)")
    values = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f):
            if lineno < (start_line + end_line) / 2:
                continue
            if lineno > end_line:
                break
            m = pattern.search(line)
            if m:
                values.append(float(m.group(1)))
            else:
                print(f"Warning: Line {lineno} has no tokens_per_second_per_gpu")

    if not values:
        raise ValueError(f"No tokens_per_second_per_gpu found in lines {start}-{end}")
    avg_tps = sum(values) / len(values) / 512
    return avg_tps


def main():
    parser = argparse.ArgumentParser(description="Parse vLLM and torchtune logs to get benchmark results")
    parser.add_argument(
        "--vllm-log",
        type=Path,
        help="Path to vLLM client log")
    parser.add_argument(
        "--torchtune-log",
        type=Path,
        help="Path to torchtune output log")
    parser.add_argument(
        "--mps-percent",
        type=int,
        help="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=2,
        help="Number of workers per node")
    parser.add_argument(
        "--start-line",
        type=int,
        help="the beginning of benchmark results")
    parser.add_argument(
        "--end-line",
        type=int,
        help="the end of benchmark results")
    parser.add_argument(
        "--result-path",
        type=str,
        default="result.txt",
        help="Path to result file")

    args = parser.parse_args()

    with open(args.result_path, 'a', encoding='utf-8') as f:
        print("{s:{c}^{n}}".format(s='', n=50, c='='), file=f)
        print(fetch_line(args.vllm_log, "P99 TTFT (ms)").rstrip('\n'), file=f)
        print(fetch_line(args.vllm_log, "P99 TPOT (ms)").rstrip('\n'), file=f)
        avg_tps = calculate_throughput(args.torchtune_log, args.start_line, args.end_line)
        print("{:<40} {:<10.2f}".format("PEFT throughput (sample/s):", avg_tps * args.nproc_per_node), file=f)
        print("{:<40} {:<10.2f}".format("PEFT CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:", args.mps_percent), file=f)

if __name__ == "__main__":
    main()
