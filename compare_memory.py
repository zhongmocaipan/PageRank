import os
import subprocess
import psutil
import time
import matplotlib.pyplot as plt

def run_and_monitor(script):
    cmd = ["python", script]
    process = subprocess.Popen(cmd)
    p = psutil.Process(process.pid)
    max_memory = 0
    try:
        while process.poll() is None:
            try:
                memory_info = p.memory_info()
                memory_used = memory_info.rss / (1024 * 1024)
                max_memory = max(max_memory, memory_used)
                time.sleep(0.1)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                break
    except KeyboardInterrupt:
        process.kill()
    process.wait()
    return max_memory

def main():
    scripts = ["main.py", "main_opt.py"]
    memory_usages = []

    for script in scripts:
        if not os.path.exists(script):
            print(f"{script} not found!")
            continue
        print(f"Running {script}...")
        mem = run_and_monitor(script)
        memory_usages.append(mem)
        print(f"{script} Max Memory: {mem:.2f} MB")

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.bar(["Original (main.py)", "Optimized (main_opt.py)"], memory_usages, color=["skyblue", "salmon"])
    plt.ylabel("Maximum Memory Usage (MB)")
    plt.title("Memory Usage Comparison")
    plt.tight_layout()
    plt.savefig("memory_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
