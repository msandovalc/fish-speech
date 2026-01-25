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

        # Model Path (Default or Custom)
        self.base_model_path = base_model_path or (self.checkpoints_dir / "openaudio-s1-mini")

        print(f"{Fore.CYAN}🚀 Initializing Kaggle Trainer for: {self.project_name}")
        print(f"   🧠 Base Model Path: {self.base_model_path}")

        self._validate_paths()

    def _validate_paths(self):
        # 1. Check Data
        if not self.data_protos.exists():
            print(f"{Fore.RED}❌ Training data not found at: {self.data_protos}")
            print(f"   👉 Run 'prepare_fish_data.py' first.")
            sys.exit(1)

        # 2. Check Model Weights (Support .safetensors, .bin, and .pth)
        has_safetensors = (self.base_model_path / "model.safetensors").exists()
        has_bin = (self.base_model_path / "pytorch_model.bin").exists()
        has_pth = (self.base_model_path / "model.pth").exists()

        if not (has_safetensors or has_bin or has_pth):
            print(f"{Fore.RED}❌ Base model weights NOT found at: {self.base_model_path}")
            print(f"   ⚠️ Expected 'model.safetensors', 'pytorch_model.bin', or 'model.pth'.")
            try:
                print(f"   📂 Files actually found: {list(self.base_model_path.glob('*'))}")
            except:
                print("   (Cannot list files)")
            sys.exit(1)

        print(f"{Fore.GREEN}   ✅ Base model validated successfully.")

    def train(self):
        print(f"{Fore.MAGENTA}🔥 Starting LoRA Fine-Tuning (Hydra Fixed)...")
        print(f"{Fore.YELLOW}⚠️  Using Tesla T4 Settings (Batch Size 4)")

        # --- ARREGLO DE HYDRA ---
        # En lugar de usar data.root (que ya no existe), apuntamos directo a proto_path

        cmd = [
            sys.executable, str(self.train_script),
            "--config-name", "text2semantic_finetune",
            f"project={self.project_name}",

            # --- RUTAS CORREGIDAS ---
            f"data.train_dataset.proto_path={self.data_protos}",
            f"data.val_dataset.proto_path={self.data_protos}",

            # Modelo base
            f"model.model.base_checkpoint_path={self.base_model_path}",
            f"trainer.default_root_dir={self.root}/results/{self.project_name}",

            # --- AJUSTES KAGGLE T4 ---
            "data.batch_size=4",
            "trainer.accumulate_grad_batches=4",
            "model.model.lora_config.r=8",
            "trainer.precision=16-mixed",
            "data.num_workers=2",
            "trainer.max_epochs=15",
            "trainer.val_check_interval=0.5",
        ]

        # Fix de entorno para PYTHONPATH
        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{str(self.root)}{os.pathsep}{current_pythonpath}"
        env["PYTHONUNBUFFERED"] = "1"

        try:
            # Ejecutar entrenamiento
            subprocess.check_call(cmd, cwd=str(self.root), env=env)
            print(f"\n{Fore.GREEN}✨ TRAINING FINISHED SUCCESSFULLY!")
            print(f"   💾 Checkpoints: {self.root}/results/{self.project_name}")

        except KeyboardInterrupt:
            print(f"\n{Fore.RED}🛑 Training stopped by user.")
        except subprocess.CalledProcessError:
            print(f"\n{Fore.RED}❌ Training failed. Check the logs above.")


if __name__ == "__main__":
    # --- AUTO-DETECT ROOT ---
    # Asumiendo que el script corre en /kaggle/working/fish-speech/train_fish_model.py
    # .parent = voices
    # .parent.parent = fish-speech (Project Root)
    PROJECT_ROOT = Path(__file__).resolve().parent

    # Nombre de tu proyecto
    PROJECT_NAME = "speaker_03_lora_v1"

    # Inicializar y Entrenar
    trainer = FishTrainer(PROJECT_ROOT, PROJECT_NAME)
    trainer.train()