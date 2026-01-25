import os
import logging
import subprocess
import sys
import torch
from pathlib import Path
from colorama import init, Fore, Style
# We use snapshot_download to get the full model folder structure
from huggingface_hub import snapshot_download

# --- SETUP ---
init(autoreset=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class FishTrainer:
    def __init__(self, project_root: Path, project_name: str, base_model_path: Path):
        self.root = project_root
        self.project_name = project_name
        self.checkpoints_dir = self.root / "checkpoints"
        self.data_protos = self.root / "fish_training_data" / "protos"
        self.train_script = self.root / "fish_speech" / "train.py"
        self.base_model_path = base_model_path

        print(f"{Fore.CYAN}🚀 Initializing Kaggle Trainer for: {self.project_name}")
        print(f"   🧠 Base Model Path: {self.base_model_path}")
        self._validate_paths()

    def _validate_paths(self):
        if not self.data_protos.exists():
            print(f"{Fore.RED}❌ Training data not found at: {self.data_protos}")
            sys.exit(1)

        if not (self.base_model_path / "model.safetensors").exists():
            if not (self.base_model_path / "pytorch_model.bin").exists():
                print(f"{Fore.RED}❌ Base model NOT found at: {self.base_model_path}")
                print(f"   ⚠️ Please check your Kaggle Input path.")
                try:
                    print(f"   📂 Files found: {list(self.base_model_path.glob('*'))}")
                except:
                    print("   (Cannot list files)")
                sys.exit(1)

        print(f"{Fore.GREEN}   ✅ Base model validated.")

    def train(self):
        print(f"{Fore.MAGENTA}🔥 Starting LoRA Fine-Tuning (Direct Input Mode)...")
        print(f"{Fore.YELLOW}⚠️  Using Tesla T4 Settings (Batch Size 4)")

        cmd = [
            sys.executable, str(self.train_script),
            "--config-name", "text2semantic_finetune",
            f"project={self.project_name}",
            f"data.root={self.data_protos}",
            f"data.val_root={self.data_protos}",
            f"model.model.base_checkpoint_path={self.base_model_path}",
            f"trainer.default_root_dir={self.root}/results/{self.project_name}",
            "data.batch_size=4",
            "trainer.accumulate_grad_batches=4",
            "model.model.lora_config.r=8",
            "trainer.precision=16-mixed",
            "data.num_workers=2",
            "trainer.max_epochs=15",
            "trainer.val_check_interval=0.5",
        ]

        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{str(self.root)}{os.pathsep}{current_pythonpath}"
        env["PYTHONUNBUFFERED"] = "1"

        try:
            subprocess.check_call(cmd, cwd=str(self.root), env=env)
            print(f"\n{Fore.GREEN}✨ TRAINING FINISHED SUCCESSFULLY!")
            print(f"   💾 Checkpoints: {self.root}/results/{self.project_name}")

        except KeyboardInterrupt:
            print(f"\n{Fore.RED}🛑 Training stopped.")
        except subprocess.CalledProcessError:
            print(f"\n{Fore.RED}❌ Training failed. Check logs.")

if __name__ == "__main__":
    # Auto-detect Project Root (Assuming this script is in /voices/)
    PROJECT_ROOT = Path(__file__).resolve().parent
    # Define your project name (Folder name for results)
    PROJECT_NAME = "speaker_03_lora_v1"
    KAGGLE_INPUT_MODEL = Path("/kaggle/working/fish-speech/checkpoints/openaudio-s1-mini")

    trainer = FishTrainer(PROJECT_ROOT, PROJECT_NAME, KAGGLE_INPUT_MODEL)
    trainer.train()