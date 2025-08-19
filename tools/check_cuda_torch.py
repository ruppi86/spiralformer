import sys
import platform
from typing import Optional


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def try_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception as exc:  # noqa: BLE001
        print(f"torch import failed: {exc}")
        return None


def describe_python_env() -> None:
    print_header("Python environment")
    print(f"python_executable= {sys.executable}")
    print(f"python_version= {platform.python_version()}")
    print(f"platform= {platform.platform()}")
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    print(f"prefix= {sys.prefix}")
    print(f"base_prefix= {base_prefix}")
    print(f"venv_active= {sys.prefix != base_prefix}")


def describe_torch(torch_module) -> None:
    print_header("PyTorch installation")
    try:
        print(f"torch_version= {torch_module.__version__}")
        cuda_version = getattr(torch_module.version, "cuda", None)
        print(f"built_with_cuda= {cuda_version}")
        print(f"cuda_available= {torch_module.cuda.is_available()}")
        if torch_module.cuda.is_available():
            print(f"device_count= {torch_module.cuda.device_count()}")
            current_index = torch_module.cuda.current_device()
            print(f"current_device_index= {current_index}")
            print(f"current_device_name= {torch_module.cuda.get_device_name(current_index)}")
            cudnn_version: Optional[int] = getattr(torch_module.backends.cudnn, "version", lambda: None)()
            print(f"cudnn_version= {cudnn_version}")
            # bf16 support can be informative for AMP settings
            bf16_supported = False
            if hasattr(torch_module.cuda, "is_bf16_supported"):
                try:
                    bf16_supported = bool(torch_module.cuda.is_bf16_supported())
                except Exception:
                    bf16_supported = False
            print(f"bf16_supported= {bf16_supported}")
        else:
            print("No CUDA device visible to PyTorch.")
    except Exception as exc:  # noqa: BLE001
        print(f"torch describe failed: {exc}")


def run_cuda_tensor_test(torch_module) -> None:
    print_header("CUDA tensor test")
    if not torch_module.cuda.is_available():
        print("SKIP: CUDA not available in this interpreter.")
        return
    try:
        device = torch_module.device("cuda")
        a = torch_module.randn((1024, 1024), device=device, dtype=torch_module.float16)
        b = torch_module.randn((1024, 1024), device=device, dtype=torch_module.float16)
        c = a @ b
        print(f"matmul_ok= True, result_shape= {tuple(c.shape)}, dtype= {c.dtype}")
        torch_module.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"matmul_ok= False, error= {exc}")


def run_amp_test(torch_module) -> None:
    print_header("AMP (autocast + GradScaler) test")
    if not torch_module.cuda.is_available():
        print("SKIP: CUDA not available in this interpreter.")
        return
    try:
        model = torch_module.nn.Linear(256, 256).to("cuda")
        opt = torch_module.optim.SGD(model.parameters(), lr=1e-3)
        scaler = torch_module.cuda.amp.GradScaler()

        x = torch_module.randn((32, 256), device="cuda")
        y = torch_module.randn((32, 256), device="cuda")

        with torch_module.cuda.amp.autocast():
            pred = model(x)
            loss = torch_module.nn.functional.mse_loss(pred, y)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch_module.cuda.synchronize()
        print("amp_ok= True")
    except Exception as exc:  # noqa: BLE001
        print(f"amp_ok= False, error= {exc}")


def main() -> None:
    describe_python_env()
    torch_module = try_import_torch()
    if torch_module is None:
        print("\nPyTorch is not installed for this Python interpreter.\n")
        return
    describe_torch(torch_module)
    run_cuda_tensor_test(torch_module)
    run_amp_test(torch_module)


if __name__ == "__main__":
    main()


