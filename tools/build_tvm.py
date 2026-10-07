#!/usr/bin/env python3
"""Apache TVM Compilation and Build Tool for PARSeq / STRHub.
Compiles exported ONNX models to optimized shared libraries (.so / .tar) using
Apache TVM (Relay / Relax) targeting CUDA (Tensor Cores / GPU) or LLVM (AVX2 / CPU).
Mirrors architectural parity with tools/build_tensorrt.py.
"""

import os
import sys

# Ensure UTF-8 output encoding across platforms
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Suppress the optional torch C dlpack JIT rebuild
os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"

import time
import json
import argparse
from typing import Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnx


def parse_target(target_str: str, allow_cpu_fallback: bool = True):
    """Resolves target string to a valid TVM Target object with fallback protection."""
    import tvm
    target_clean = target_str.strip()

    # Determine if cuda target requested
    if target_clean.startswith("cuda"):
        if not tvm.cuda().exist:
            if allow_cpu_fallback:
                print(f"[Aviso] TVM CUDA runtime não está ativo neste ambiente. Alternando para LLVM CPU.")
                return tvm.target.Target("llvm")
            else:
                raise RuntimeError("Target CUDA requisitado, mas TVM não foi compilado com suporte a CUDA.")
        try:
            return tvm.target.Target(target_clean)
        except Exception:
            return tvm.target.Target("cuda")
    else:
        try:
            return tvm.target.Target(target_clean)
        except Exception:
            return tvm.target.Target("llvm")


def build_tvm_library(
    onnx_path: str,
    output_path: str,
    target: str = "cuda",
    precision: str = "fp32",
    batch_size: int = 1,
    img_size: Tuple[int, int] = (32, 128),
    opt_level: int = 3,
    tune: bool = False,
    tune_trials: int = 200,
) -> Dict[str, Any]:
    """Compiles an ONNX model into an Apache TVM native library (.so / .tar)."""
    import tvm

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Modelo ONNX de entrada não encontrado: {onnx_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    t_start = time.perf_counter()

    print(f"\n==================================================")
    print(f"Compilando TVM: {onnx_path} -> {output_path}")
    print(f"Alvo: {target} | Precisão: {precision.upper()} | Lote: {batch_size} | Opt: {opt_level}")
    print(f"==================================================")

    # 1. Carregar modelo ONNX
    onnx_model = onnx.load(onnx_path)
    shape_dict = {"images": (batch_size, 3, img_size[0], img_size[1])}
    precision = precision.lower().strip()
    tvm_target = parse_target(target)

    compiled_lib = None
    params_dict = None
    graph_json = None
    backend_mode = "relay" if hasattr(tvm, "relay") else "relax"

    if hasattr(tvm, "relay"):
        from tvm import relay
        print("Frontend TVM selecionado: Relay (Graph IR)")
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

        # 3. Passes de precisão e transformação
        if precision == "fp16":
            print("Aplicando Relay ToMixedPrecision (float16)...")
            try:
                from tvm.relay.transform import ToMixedPrecision
                mod = ToMixedPrecision("float16")(mod)
            except Exception as e:
                print(f"Aviso ao converter ToMixedPrecision: {e}")
        elif precision == "int8":
            print("Aplicando otimizações e inferência de tipo para INT8 no Relay...")
            try:
                seq = tvm.transform.Sequential([
                    relay.transform.InferType(),
                    relay.transform.SimplifyInference(),
                    relay.transform.FoldConstant(),
                ])
                mod = seq(mod)
            except Exception as e:
                print(f"Aviso ao otimizar INT8 no Relay: {e}")

        # 4. Auto-tuning opcional
        if tune:
            records_dir = os.path.dirname(os.path.abspath(output_path))
            records_file = os.path.join(records_dir, "tuning_records.json")
            print(f"Executando auto-tuning ({tune_trials} trials) salvando em {records_file}...")
            try:
                from tvm import auto_scheduler
                tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target=tvm_target)
                tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
                tune_opt = auto_scheduler.TuningOptions(
                    num_measure_trials=tune_trials,
                    runner=auto_scheduler.LocalRunner(repeat=1, number=1, min_repeat_ms=20),
                    measure_callbacks=[auto_scheduler.RecordToFile(records_file)],
                )
                tuner.tune(tune_opt)
            except Exception as te:
                print(f"Aviso: falha durante auto-tuning: {te}. Prosseguindo com compilação direta.")

        # 5. Build com Relay
        with tvm.transform.PassContext(opt_level=opt_level):
            compiled_lib = relay.build(mod, target=tvm_target, params=params)

    elif hasattr(tvm, "relax"):
        from tvm import relax
        print("Frontend TVM selecionado: Relax (TVM Unity IR)")
        from tvm.relax.frontend.onnx import from_onnx
        mod = from_onnx(onnx_model, shape_dict)

        # Passes de otimização no Relax
        if precision == "fp16":
            try:
                mod = relax.transform.ToMixedPrecision("float16")(mod)
            except Exception:
                pass

        if tune:
            records_dir = os.path.dirname(os.path.abspath(output_path))
            records_file = os.path.join(records_dir, "tuning_records.json")
            print(f"Executando tuning Relax/MetaSchedule em {records_file}...")
            try:
                from tvm import meta_schedule as ms
                db = ms.tune_tir(
                    mod=mod,
                    target=tvm_target,
                    work_dir=records_dir,
                    max_trials_global=tune_trials,
                )
                mod = ms.apply_history_best(db, mod)
            except Exception as me:
                print(f"Aviso MetaSchedule: {me}")

        compiled_lib = relax.build(mod, target=tvm_target)
    else:
        raise RuntimeError("Nenhum frontend TVM (nem Relay nem Relax) encontrado na instalação atual.")

    # 6. Exportar runtime serializado
    print(f"Exportando biblioteca serializada para: {output_path}...")
    compiled_lib.export_library(output_path)

    t_end = time.perf_counter()
    build_time_s = t_end - t_start
    lib_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    # Salvar metadados em manifesto JSON acompanhante
    manifest_path = output_path + ".manifest.json"
    manifest_data = {
        "onnx_path": onnx_path,
        "output_path": output_path,
        "target": str(tvm_target),
        "target_raw": target,
        "precision": precision,
        "batch_size": batch_size,
        "img_size": list(img_size),
        "opt_level": opt_level,
        "backend_mode": backend_mode,
        "tuned": tune,
        "build_time_seconds": round(build_time_s, 2),
        "lib_size_mb": round(lib_size_mb, 2),
        "tvm_version": tvm.__version__,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Compilação TVM concluída em {build_time_s:.2f}s! Tamanho: {lib_size_mb:.2f} MB")
    print(f"Manifesto salvo em: {manifest_path}")

    return manifest_data


def build_all_tvm_libraries(
    target: str = "cuda",
    batch_size: int = 1,
    opt_level: int = 3
) -> Dict[str, Any]:
    """Compila em lote todas as variantes PARSeq M0..M6 existentes para TVM."""
    targets = [
        ("onnx/parseq_m0_ar_fp32.onnx", "tvm_lib/parseq_m0_ar_fp32.so", "fp32"),
        ("onnx/parseq_m1_nar_fp32.onnx", "tvm_lib/parseq_m1_nar_fp32.so", "fp32"),
        ("onnx/parseq_m2_nar_fp16.onnx", "tvm_lib/parseq_m2_nar_fp16.so", "fp16"),
        ("onnx/parseq_m3_nar_int8_naive.onnx", "tvm_lib/parseq_m3_nar_int8_naive.so", "int8"),
        ("onnx/parseq_m4_nar_int8_ptq.onnx", "tvm_lib/parseq_m4_nar_int8_ptq.so", "int8"),
        ("onnx/parseq_m5_nar_int8_io_ptq.onnx", "tvm_lib/parseq_m5_nar_int8_io_ptq.so", "int8"),
        ("onnx/parseq_m6_nar_int8_io_qat.onnx", "tvm_lib/parseq_m6_nar_int8_io_qat.so", "int8"),
        ("onnx/parseq_m6_ibert.onnx", "tvm_lib/parseq_m6_ibert.so", "int8"),
        ("onnx/parseq_m6_ivit.onnx", "tvm_lib/parseq_m6_ivit.so", "int8"),
    ]

    results = {}
    for onnx_p, out_p, prec in targets:
        if not os.path.exists(onnx_p):
            print(f"Ignorando {onnx_p} (arquivo não encontrado no diretório onnx/).")
            continue
        try:
            res = build_tvm_library(
                onnx_path=onnx_p,
                output_path=out_p,
                target=target,
                precision=prec,
                batch_size=batch_size,
                opt_level=opt_level
            )
            results[out_p] = res
        except Exception as e:
            print(f"Erro ao compilar {out_p}: {e}")
            results[out_p] = f"FAILED: {e}"

    print("\n=== Resumo do Build Apache TVM ===")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"  {os.path.basename(k)}: {v['lib_size_mb']:.2f} MB em {v['build_time_seconds']:.1f}s")
        else:
            print(f"  {os.path.basename(k)}: {v}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apache TVM Compiler for PARSeq models")
    parser.add_argument("--onnx", type=str, default=None, help="Caminho do modelo .onnx")
    parser.add_argument("--output", type=str, default=None, help="Caminho da biblioteca .so / .tar")
    parser.add_argument("--target", type=str, default="cuda", help="Target TVM (e.g. cuda, llvm)")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--batch_size", type=int, default=1, help="Dimensão do batch de inferência")
    parser.add_argument("--img_size", type=int, nargs=2, default=[32, 128], help="Shape da imagem (H W)")
    parser.add_argument("--opt_level", type=int, default=3, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--tune", action="store_true", help="Acionar auto-tuning")
    parser.add_argument("--tune_trials", type=int, default=200, help="Número de medições no auto-tuning")
    parser.add_argument("--all", action="store_true", help="Compilar todas as variantes disponíveis")
    args = parser.parse_args()

    if args.all:
        build_all_tvm_libraries(target=args.target, batch_size=args.batch_size, opt_level=args.opt_level)
    else:
        if not args.onnx or not args.output:
            parser.error("--onnx e --output são obrigatórios a menos que --all seja informado.")
        build_tvm_library(
            onnx_path=args.onnx,
            output_path=args.output,
            target=args.target,
            precision=args.precision,
            batch_size=args.batch_size,
            img_size=tuple(args.img_size),
            opt_level=args.opt_level,
            tune=args.tune,
            tune_trials=args.tune_trials,
        )
