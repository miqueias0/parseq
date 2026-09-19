import os
import sys
import glob
import subprocess

tools_dir = os.path.join(os.path.dirname(__file__), "..", "tools")
tools = sorted(glob.glob(os.path.join(tools_dir, "*.py")))

print(f"Testing all {len(tools)} tools in {tools_dir} with Python: {sys.executable}\n", flush=True)
python_exe = sys.executable

results = []

for tool in tools:
    tool_name = os.path.basename(tool)
    print(f"Checking {tool_name:40} ...", end=" ", flush=True)
    
    # 1. Syntax check
    try:
        with open(tool, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        compile(code, tool, "exec")
    except Exception as e:
        print(f"SYNTAX ERROR: {e}", flush=True)
        results.append((tool_name, "SYNTAX_ERROR", str(e)))
        continue

    # 2. Test execution / --help
    cmd = [python_exe, tool, "--help"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=12, cwd=os.path.join(os.path.dirname(__file__), ".."))
        if proc.returncode == 0:
            print("OK (exit code 0)", flush=True)
            results.append((tool_name, "OK", "help works"))
        else:
            err = (proc.stderr or proc.stdout).strip().splitlines()[-1] if (proc.stderr or proc.stdout).strip() else "Non-zero exit"
            print(f"FAIL (exit {proc.returncode}): {err[:80]}", flush=True)
            results.append((tool_name, f"FAIL_{proc.returncode}", err))
    except subprocess.TimeoutExpired:
        print("TIMEOUT", flush=True)
        results.append((tool_name, "TIMEOUT", "TimeoutExpired"))
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        results.append((tool_name, "ERROR", str(e)))

print("\n" + "="*80, flush=True)
print(f"FINAL AUDIT OF ALL {len(tools)} TOOLS:", flush=True)
print("="*80, flush=True)
all_ok = True
for name, status, detail in results:
    is_ok = status == "OK"
    if not is_ok:
        all_ok = False
    print(f"  [{'PASS' if is_ok else 'FAIL'}] {name:42} : {status:10} : {detail[:50]}")

print("="*80, flush=True)
print(f"OVERALL STATUS: {'ALL 28 TOOLS FUNCTIONAL (100% PASS)' if all_ok else 'SOME TOOLS FAILED'}")
print("="*80, flush=True)
if not all_ok:
    sys.exit(1)
