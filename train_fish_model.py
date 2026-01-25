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
        """Ensures that the protobuf data exists before trying to train."""
        if not self.data_protos.exists():
            print(f"{Fore.RED}❌ Training data not found at: {self.data_protos}")
            print(f"{Fore.YELLOW}   👉 Did you run 'prepare_fish_data.py' successfully?")
            sys.exit(1)

    def _ensure_model_weights(self):
        """
        Downloads the full LLAMA model (OpenAudio S1 Mini) directly into the
        project's 'checkpoints' folder.
        """
        print(f"{Fore.YELLOW}🔍 Checking Base Model weights...")

        repo_id = "fishaudio/openaudio-s1-mini"

        # Check if critical files exist to avoid re-downloading unnecessary data
        if (self.base_model_path / "model.safetensors").exists():
            print(f"{Fore.GREEN}   ✅ Base model found at: {self.base_model_path}")
            return

        print(f"{Fore.CYAN}   ⬇️  Downloading model to {self.base_model_path}...")
        print(f"       (This may take a while, ~800MB - 1.5GB)")

        try:
            # snapshot_download ensures we get config.json, tokenizer, weights, etc.
            snapshot_download(
                repo_id=repo_id,
                local_dir=self.base_model_path,
                local_dir_use_symlinks=False  # Force actual files for Windows stability
            )
            print(f"{Fore.GREEN}   ✅ Download complete.")
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to download model: {e}")
            sys.exit(1)

    def train(self):
        """
        Executes the training process using subprocess.
        Optimized for Low VRAM (4GB).
        """
        # 1. Ensure weights are present
        self._ensure_model_weights()

        print(f"{Fore.MAGENTA}🔥 Starting LoRA Fine-Tuning...")
        print(f"{Fore.YELLOW}⚠️  OPTIMIZED FOR 4GB VRAM (Quadro T1000)")
        print(f"   - Batch Size: 1 (Prevents OOM)")
        print(f"   - Gradient Accumulation: 16 (Simulates Batch 16)")

        # 2. Construct the training command
        # We override specific Hydra configurations via CLI arguments
        cmd = [
            sys.executable, str(self.train_script),
            "--config-name", "text2semantic_finetune",
            f"project={self.project_name}",

            # --- PATH CONFIGURATION ---
            f"data.root={self.data_protos}",
            f"data.val_root={self.data_protos}",  # Use same data for validation for simplicity
            f"model.model.base_checkpoint_path={self.base_model_path}",
            f"trainer.default_root_dir={self.root}/results/{self.project_name}",

            # --- LOW VRAM OPTIMIZATIONS (CRITICAL FOR 4GB) ---
            "data.batch_size=1",  # Lowest possible batch size
            "trainer.accumulate_grad_batches=16",  # High accumulation to compensate for low batch
            "model.model.lora_config.r=8",  # Keep LoRA rank low (8 is standard/light)
            "trainer.precision=16-mixed",  # Use FP16 to save memory
            "data.num_workers=0",  # 0 is safer on Windows to avoid spawning overhead

            # --- TRAINING DURATION ---
            "trainer.max_epochs=10",  # Adjust as needed
            "trainer.val_check_interval=0.5",  # Validate less often
        ]

        # 3. Windows-specific fix for Multi-GPU/Process communication
        if os.name == 'nt':
            cmd.append("trainer.strategy.process_group_backend=gloo")

        # 4. Run the command
        try:
            # Copy environment and force unbuffered output for real-time logs
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            subprocess.check_call(cmd, cwd=str(self.root), env=env)

            print(f"\n{Fore.GREEN}✨ TRAINING FINISHED SUCCESSFULLY!")
            print(f"   💾 Checkpoints located at: {self.root}/results/{self.project_name}")

        except KeyboardInterrupt:
            print(f"\n{Fore.RED}🛑 Training stopped by user.")
        except subprocess.CalledProcessError:
            print(f"\n{Fore.RED}❌ Training failed. Please check the logs above for CUDA OOM errors.")


if __name__ == "__main__":
    # Auto-detect Project Root (Assuming this script is in /voices/)
    PROJECT_ROOT = Path(__file__).resolve().parent
    # Define your project name (Folder name for results)
    PROJECT_NAME = "speaker_03_lora_v1"
    KAGGLE_INPUT_MODEL = Path("/kaggle/input/openaudio-s1-min")

    trainer = FishTrainer(PROJECT_ROOT, PROJECT_NAME, KAGGLE_INPUT_MODEL)
    trainer.train()