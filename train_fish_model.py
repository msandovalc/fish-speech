import os
import logging
import subprocess
import sys
import torch
from pathlib import Path
from colorama import init, Fore, Style

# --- SETUP ---
init(autoreset=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class FishTrainer:
    def __init__(self, project_root: Path, project_name: str, base_model_path: Path = None):
        self.root = project_root
        self.project_name = project_name
        self.checkpoints_dir = self.root / "checkpoints"
        self.data_protos = self.root / "fish_training_data" / "protos"
        self.train_script = self.root / "fish_speech" / "train.py"
        self.base_model_path = base_model_path or (self.checkpoints_dir / "openaudio-s1-mini")

        print(f"{Fore.CYAN}🚀 RunPod Stable Trainer: {self.project_name}")
        self._validate_paths()

    def _validate_paths(self):
        if not self.data_protos.exists():
            print(f"{Fore.RED}❌ Data not found at: {self.data_protos}")
            sys.exit(1)
        valid_exts = ["model.safetensors", "pytorch_model.bin", "model.pth"]
        if not any((self.base_model_path / ext).exists() for ext in valid_exts):
            print(f"{Fore.RED}❌ Base model weights NOT found.")
            sys.exit(1)

    def train(self):
        import torch, os, subprocess, sys
        torch.cuda.empty_cache()

        abs_root = "/workspace/fish-speech/results/camila_voice_v1_stable"
        os.makedirs(abs_root, exist_ok=True)

        print(f"🚀 MODO 'BALA': 1.15 it/s + Saltando validación")
        print(f"📁 Guardando en: {abs_root}/lightning_logs/version_X/checkpoints/")

        cmd = [
            sys.executable, str(self.train_script),
            "--config-name", "text2semantic_finetune",
            f"project={self.project_name}",
            f"train_dataset.proto_files=['{str(self.data_protos)}']",
            f"val_dataset.proto_files=['{str(self.data_protos)}']",
            f"pretrained_ckpt_path={str(self.base_model_path)}",
            "+lora@model.model.lora_config=r_8_alpha_16",

            # PARÁMETROS DE VELOCIDAD ÓPTIMA
            "data.batch_size=1",
            "trainer.devices=1",
            "++trainer.accumulate_grad_batches=16",
            "++trainer.precision=bf16-mixed",
            "++trainer.max_steps=5000",

            # NO VALIDACIÓN = MÁXIMA VELOCIDAD
            "++trainer.val_check_interval=1000",
            "++trainer.limit_val_batches=0",
            "++trainer.num_sanity_val_steps=0",

            # GUARDADO FORZADO (SINTAXIS CORREGIDA)
            f"trainer.default_root_dir={abs_root}",
            "++callbacks.model_checkpoint.every_n_train_steps=250",
            "++callbacks.model_checkpoint.save_top_k=-1",
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{str(self.root)}{os.pathsep}{env.get('PYTHONPATH', '')}"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        try:
            subprocess.check_call(cmd, cwd=str(self.root), env=env)
            print("✅ VERIFICANDO:")
            os.system(f"find {abs_root} -name '*.ckpt'")
            os.system(f"ls -la {abs_root}/lightning_logs/")
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    PROJECT_ROOT = Path("/workspace/fish-speech")
    PROJECT_NAME = "camila_voice_v1_stable"
    trainer = FishTrainer(PROJECT_ROOT, PROJECT_NAME)
    trainer.train()