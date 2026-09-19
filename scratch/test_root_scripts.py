import subprocess
import sys

root_scripts = ['read.py', 'test.py', 'train.py', 'tune.py', 'bench.py', 'benchmark_fps.py']
for s in root_scripts:
    cmd = [sys.executable, s, '--help']
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        print(f"{s:20} -> Exit: {proc.returncode} | Output: {proc.stdout[:60].strip() if proc.stdout else proc.stderr[:60].strip()}")
    except Exception as e:
        print(f"{s:20} -> Error: {e}")
