import argparse
from pathlib import Path
import re


def fetch_line(filepath, keyword):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if keyword in line:
                return line


def calculate_throughput(filepath, start_line, end_line):
    pattern = re.compile(r"finetune:\s*([0-9]+(?:\.[0-9]+)?)")
    pattern2 = re.compile(r"Duration\(s\):\s*([0-9]+(?:\.[0-9]+)?)")
    values = []
    durations = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f):
            if lineno < start_line + 2:
                continue
            if lineno > end_line - 2:
                break

            m = pattern.search(line)
            if m:
                values.append(float(m.group(1)))
            else:
                print(f"Warning: Line {lineno} has no finetune")

            m2 = pattern2.search(line)
            if m2:
                durations.append(float(m2.group(1)))
            else:
                print(f"Warning: Line {lineno} has no durations")

    if not values or not durations:
        raise ValueError(f"No finetune/durations found in the file")

    avg_tps = sum(x * y for x, y in zip(values, durations)) / sum(durations)
    return avg_tps


def main():
    parser = argparse.ArgumentParser(description="Parse vLLM and torchtune logs to get benchmark results")
    parser.add_argument(
        "--vllm-log",
        type=Path,
        help="Path to vLLM client log")
    parser.add_argument(
        "--fineinfer-log",
        type=Path,
        help="Path to fineinfer output log")
    parser.add_argument(
        "--fineinfer-defer",
        type=float,
        help="defer in seconds before new prefill")
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
        avg_tps = calculate_throughput(args.fineinfer_log, args.start_line, args.end_line)
        print("{:<40} {:<10.2f}".format("PEFT throughput (sample/s):", avg_tps), file=f)
        print("{:<40} {:<10.2f}".format("FineInfer defer (ms):", args.fineinfer_defer * 1000), file=f)

if __name__ == "__main__":
    main()
