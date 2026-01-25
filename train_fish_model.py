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

        # Paths ajustados a tu estructura en RunPod
        self.checkpoints_dir = self.root / "checkpoints"
        self.data_protos = self.root / "fish_training_data" / "protos"
        self.train_script = self.root / "fish_speech" / "train.py"

        # Model Path: Apuntamos a la carpeta del modelo que descargamos
        self.base_model_path = base_model_path or (self.checkpoints_dir / "openaudio-s1-mini")

        print(f"{Fore.CYAN}🚀 Initializing RunPod Trainer for: {self.project_name}")
        print(f"   🧠 Base Model Path: {self.base_model_path}")

        self._validate_paths()

    def _validate_paths(self):
        if not self.data_protos.exists():
            print(f"{Fore.RED}❌ Training data not found at: {self.data_protos}")
            sys.exit(1)

        # Validar existencia de pesos (model.pth es el que bajamos de HF)
        valid_exts = ["model.safetensors", "pytorch_model.bin", "model.pth"]
        if not any((self.base_model_path / ext).exists() for ext in valid_exts):
            print(f"{Fore.RED}❌ Base model weights NOT found at: {self.base_model_path}")
            sys.exit(1)

        print(f"{Fore.GREEN}   ✅ Base model and data validated.")

    def train(self):
        torch.cuda.empty_cache()

        print(f"{Fore.MAGENTA}🔥 Starting LoRA Fine-Tuning (RTX 4090 OPTIMIZED)...")

        # Construcción del comando respetando la sintaxis de Hydra (+)
        cmd = [
            sys.executable, str(self.train_script),
            "--config-name", "text2semantic_finetune",
            f"project={self.project_name}",

            # --- DATASET ---
            f"train_dataset.proto_files=['{str(self.data_protos)}']",
            f"val_dataset.proto_files=['{str(self.data_protos)}']",

            # --- MODELO ---
            # Fish Speech puede aceptar la carpeta o el archivo .pth directo
            f"pretrained_ckpt_path={str(self.base_model_path / 'model.pth')}",
            f"trainer.default_root_dir={self.root}/results/{self.project_name}",

            # --- LORA ---
            "+lora@model.model.lora_config=r_8_alpha_16",

            # --- OPTIMIZACIÓN RUNPOD (v6.2) ---
            "data.batch_size=4",  # Subimos de 1 a 4 gracias a los 24GB de VRAM
            "trainer.devices=1",
            "trainer.accumulate_grad_batches=4",  # Mantenemos batch efectivo de 16 (4x4)

            # BF16 es superior en RTX serie 4000
            "trainer.precision=bf16-mixed",

            "data.num_workers=4",  # CPUs de RunPod son más rápidas

            # --- DURACIÓN (Corregido con el signo +) ---
            "+trainer.max_epochs=20",
            "trainer.val_check_interval=50",
        ]

        # Environment Fix
        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{str(self.root)}{os.pathsep}{current_pythonpath}"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        try:
            subprocess.check_call(cmd, cwd=str(self.root), env=env)
            print(f"\n{Fore.GREEN}✨ TRAINING FINISHED SUCCESSFULLY!")
            print(f"   💾 Checkpoints: {self.root}/results/{self.project_name}")

        except KeyboardInterrupt:
            print(f"\n{Fore.RED}🛑 Training stopped by user.")
        except subprocess.CalledProcessError as e:
            print(f"\n{Fore.RED}❌ Training failed. Exit code: {e.returncode}")


if __name__ == "__main__":
    # Forzamos la ruta de RunPod
    PROJECT_ROOT = Path("/workspace/fish-speech")
    PROJECT_NAME = "camila_voice_runpod_v1"
    trainer = FishTrainer(PROJECT_ROOT, PROJECT_NAME)
    trainer.train()