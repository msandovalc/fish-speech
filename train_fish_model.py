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

        # Define paths relative to the project root
        self.checkpoints_dir = self.root / "checkpoints"
        self.data_protos = self.root / "fish_training_data" / "protos"
        self.train_script = self.root / "fish_speech" / "train.py"

        # Default path to the S1-Mini model (~3.36 GB).
        # Falls back to the standard checkpoints directory if not provided.
        self.base_model_path = base_model_path or (self.checkpoints_dir / "openaudio-s1-mini")

        print(f"{Fore.CYAN}🚀 Initializing Kaggle Trainer for: {self.project_name}")
        print(f"   🧠 Base Model Path: {self.base_model_path}")

        self._validate_paths()

    def _validate_paths(self):
        """
        Ensures that training data exists and that the base model
        contains valid weight files (safetensors, bin, or pth).
        """
        # 1. Check Data
        if not self.data_protos.exists():
            print(f"{Fore.RED}❌ Training data not found at: {self.data_protos}")
            print(f"   👉 Run 'prepare_fish_data.py' first.")
            sys.exit(1)

        # 2. Check Model Weights
        # We check for multiple formats because the model might be .safetensors, .bin, or .pth
        has_safetensors = (self.base_model_path / "model.safetensors").exists()
        has_bin = (self.base_model_path / "pytorch_model.bin").exists()
        has_pth = (self.base_model_path / "model.pth").exists()  # <--- ADDED THIS

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
        """
        Executes the training subprocess with Kaggle-optimized settings.
        """
        print(f"{Fore.MAGENTA}🔥 Starting LoRA Fine-Tuning (Direct Input Mode)...")
        print(f"{Fore.YELLOW}⚠️  Using Tesla T4 Settings (Batch Size 4)")

        # Construct the command
        cmd = [
            sys.executable, str(self.train_script),
            "--config-name", "text2semantic_finetune",
            f"project={self.project_name}",

            # Paths
            f"data.root={self.data_protos}",
            f"data.val_root={self.data_protos}",
            f"model.model.base_checkpoint_path={self.base_model_path}",
            f"trainer.default_root_dir={self.root}/results/{self.project_name}",

            # --- Kaggle T4 Optimizations ---
            "data.batch_size=4",  # Balanced for T4 VRAM
            "trainer.accumulate_grad_batches=4",  # Effective batch size = 16
            "model.model.lora_config.r=8",  # Standard rank
            "trainer.precision=16-mixed",  # FP16 to save memory
            "data.num_workers=2",  # Use Kaggle CPUs
            "trainer.max_epochs=15",  # Standard run
            "trainer.val_check_interval=0.5",  # Check validation twice per epoch
        ]

        # Environment Fix: Add project root to PYTHONPATH so 'fish_speech' module is found
        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{str(self.root)}{os.pathsep}{current_pythonpath}"
        env["PYTHONUNBUFFERED"] = "1"

        try:
            subprocess.check_call(cmd, cwd=str(self.root), env=env)
            print(f"\n{Fore.GREEN}✨ TRAINING FINISHED SUCCESSFULLY!")
            print(f"   💾 Checkpoints: {self.root}/results/{self.project_name}")

        except KeyboardInterrupt:
            print(f"\n{Fore.RED}🛑 Training stopped by user.")
        except subprocess.CalledProcessError:
            print(f"\n{Fore.RED}❌ Training failed. Check the logs above.")


if __name__ == "__main__":
    # --- AUTO-DETECT ROOT ---
    # If this file is in /kaggle/working/fish-speech/train_fish_model.py
    # .parent = voices
    # .parent.parent = fish-speech (The Project Root)
    PROJECT_ROOT = Path(__file__).resolve().parent

    # Define your project name (Folder name for results)
    PROJECT_NAME = "speaker_03_lora_v1"

    # Initialize and Train
    trainer = FishTrainer(PROJECT_ROOT, PROJECT_NAME)
    trainer.train()