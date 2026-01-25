import os
import logging
import subprocess
import sys
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

        # Paths
        self.checkpoints_dir = self.root / "checkpoints"
        self.data_protos = self.root / "fish_training_data" / "protos"
        self.train_script = self.root / "fish_speech" / "train.py"

        # Model Path
        self.base_model_path = base_model_path or (self.checkpoints_dir / "openaudio-s1-mini")

        print(f"{Fore.CYAN}🚀 Initializing Kaggle Trainer for: {self.project_name}")
        print(f"   🧠 Base Model Path: {self.base_model_path}")

        self._validate_paths()

    def _validate_paths(self):
        if not self.data_protos.exists():
            print(f"{Fore.RED}❌ Training data not found at: {self.data_protos}")
            sys.exit(1)

        valid_exts = ["model.safetensors", "pytorch_model.bin", "model.pth"]
        if not any((self.base_model_path / ext).exists() for ext in valid_exts):
            print(f"{Fore.RED}❌ Base model weights NOT found at: {self.base_model_path}")
            sys.exit(1)

        print(f"{Fore.GREEN}   ✅ Base model validated.")

    def train(self):
        # Forzar limpieza de memoria antes de empezar
        torch.cuda.empty_cache()

        print(f"{Fore.MAGENTA}🔥 Starting LoRA Fine-Tuning (ULTRA-SAFE Mode)...")
        print(f"{Fore.YELLOW}⚠️  Batch Size: 1 | Accumulation: 16")

        cmd = [
            sys.executable, str(self.train_script),
            "--config-name", "text2semantic_finetune",
            f"project={self.project_name}",

            # --- DATASET ---
            f"train_dataset.proto_files=['{str(self.data_protos)}']",
            f"val_dataset.proto_files=['{str(self.data_protos)}']",

            # --- MODELO ---
            f"pretrained_ckpt_path={str(self.base_model_path)}",
            f"trainer.default_root_dir={self.root}/results/{self.project_name}",

            # --- LORA ---
            "+lora@model.model.lora_config=r_8_alpha_16",

            # --- AJUSTES CRÍTICOS DE MEMORIA (v5.9) ---
            "data.batch_size=1",  # <--- EL MÍNIMO ABSOLUTO
            "trainer.accumulate_grad_batches=16",  # <--- COMPENSACIÓN ALTA

            "trainer.precision=16-mixed",  # Ahorra VRAM
            "data.num_workers=2",  # CPUs de Kaggle

            # Epochs y validación
            "+trainer.max_epochs=15",
            "trainer.val_check_interval=100",  # Validar menos frecuente para no estresar la GPU
        ]

        # Environment Fix + Memory Fixes
        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{str(self.root)}{os.pathsep}{current_pythonpath}"
        env["PYTHONUNBUFFERED"] = "1"

        # Variable mágica para reducir fragmentación de memoria en PyTorch
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        try:
            subprocess.check_call(cmd, cwd=str(self.root), env=env)
            print(f"\n{Fore.GREEN}✨ TRAINING FINISHED SUCCESSFULLY!")
            print(f"   💾 Checkpoints: {self.root}/results/{self.project_name}")

        except KeyboardInterrupt:
            print(f"\n{Fore.RED}🛑 Training stopped by user.")
        except subprocess.CalledProcessError:
            print(f"\n{Fore.RED}❌ Training failed. Check logs.")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent
    PROJECT_NAME = "speaker_03_lora_v1"
    trainer = FishTrainer(PROJECT_ROOT, PROJECT_NAME)
    trainer.train()