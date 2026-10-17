"""Verifie que torch nightly cu128 voit bien la RTX 5080 (sm_120)."""
import sys

import torch

print(f"Python        : {sys.version.split()[0]}")
print(f"torch         : {torch.__version__}")
print(f"CUDA (build)  : {torch.version.cuda}")
print(f"CUDA dispo    : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    sys.exit("ERREUR : CUDA indisponible. Verifie le pilote NVIDIA et l'install nightly cu128.")

if "+cu128" not in torch.__version__:
    print("ATTENTION : torch n'est pas la build cu128. Un paquet l'a peut-etre remplace.")
    print("Relance : pip install --pre --force-reinstall torch torchvision "
          "--index-url https://download.pytorch.org/whl/nightly/cu128")

props = torch.cuda.get_device_properties(0)
print(f"GPU           : {props.name}")
print(f"VRAM          : {props.total_memory / 1024**3:.1f} Go")
print(f"Capability    : sm_{props.major}{props.minor}")
print(f"Archs build   : {torch.cuda.get_arch_list()}")
print(f"bf16 supporte : {torch.cuda.is_bf16_supported()}")

x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print(f"Test matmul bf16 OK : {tuple(y.shape)}")
