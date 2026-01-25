import os
import logging
import subprocess
import sys
import shutil
import torch
from pathlib import Path
from colorama import init, Fore, Style
from huggingface_hub import hf_hub_download  # Requires: pip install huggingface_hub

# --- SETUP ---
init(autoreset=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('fish_preparation.log', encoding='utf-8')]
)
logger = logging.getLogger(__name__)


class FishDataBuilder:
    def __init__(self,
                 dataset_root: Path,
                 fish_speech_root: Path,
                 checkpoints_dir: Path,
                 output_dir: Path):

        self.dataset_root = dataset_root
        self.fish_root = fish_speech_root
        self.checkpoints_dir = checkpoints_dir
        self.output_dir = output_dir

        # Paths based on OFFICIAL DOCS
        self.extract_script = self.fish_root / "tools" / "vqgan" / "extract_vq.py"
        self.build_script = self.fish_root / "tools" / "llama" / "build_dataset.py"

        # Creates output folder
        self.proto_output = self.output_dir / "protos"
        self.proto_output.mkdir(parents=True, exist_ok=True)

        self._log(f"{Fore.CYAN}🚀 Initializing Fish Data Builder (v5.0 Official Docs)...")
        self._validate_paths()

    def _log(self, message: str, level="info"):
        print(message)
        if level == "info":
            logger.info(message)
        elif level == "error":
            logger.error(message)

    def _validate_paths(self):
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"{Fore.RED}❌ Dataset not found at: {self.dataset_root}")
        if not self.fish_root.exists():
            raise FileNotFoundError(f"{Fore.RED}❌ Repo Root not found at: {self.fish_root}")
        if not self.extract_script.exists():
            raise FileNotFoundError(f"{Fore.RED}❌ Script 'extract_vq.py' not found in tools/vqgan/")

    def _check_gpu_status(self):
        self._log(f"{Fore.YELLOW}🔍 Checking Hardware Acceleration...")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            self._log(f"{Fore.GREEN}   ✅ CUDA DETECTED: {gpu_name} ({vram:.1f} GB VRAM)")
        else:
            self._log(f"{Fore.RED}   ⚠️ NO GPU DETECTED. Running on CPU (Very Slow).")

    def _ensure_checkpoint(self):
        """
        Downloads the OFFICIAL checkpoint: fishaudio/openaudio-s1-mini/codec.pth
        This matches 'modded_dac_vq' config (1024 dimensions).
        """
        model_id = "fishaudio/openaudio-s1-mini"
        filename = "codec.pth"

        # Expected path: checkpoints/openaudio-s1-mini/codec.pth
        local_model_path = self.checkpoints_dir / "openaudio-s1-mini" / filename

        if not local_model_path.exists():
            self._log(f"{Fore.YELLOW}⚠️  Official checkpoint not found. Downloading '{filename}'...")
            try:
                # Use HuggingFace Hub to download
                downloaded_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    local_dir=self.checkpoints_dir / "openaudio-s1-mini"
                )
                self._log(f"{Fore.GREEN}   ✅ Downloaded: {downloaded_path}")
                return Path(downloaded_path)
            except Exception as e:
                self._log(f"{Fore.RED}❌ Download failed: {e}", level="error")
                sys.exit(1)
        else:
            self._log(f"{Fore.GREEN}   ✅ Checkpoint found: {local_model_path}")
            return local_model_path

    def _run_subprocess_interactive(self, cmd, description):
        self._log(f"{Fore.MAGENTA}⚙️  Running: {description}...")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            subprocess.check_call(cmd, cwd=str(self.fish_root), env=env)
            self._log(f"{Fore.GREEN}✅ {description} finished successfully.")
        except subprocess.CalledProcessError as e:
            self._log(f"{Fore.RED}❌ {description} Failed with exit code {e.returncode}.", level="error")
            sys.exit(1)
        except Exception as e:
            self._log(f"{Fore.RED}❌ Execution error: {e}", level="error")
            sys.exit(1)

    def convert_txt_to_lab(self):
        self._log(f"{Fore.YELLOW}🔄 Normalizing extensions (.txt -> .lab)...")
        count = 0
        for txt_file in self.dataset_root.rglob("*.txt"):
            lab_file = txt_file.with_suffix(".lab")
            if not lab_file.exists():
                try:
                    shutil.copy2(txt_file, lab_file)
                    count += 1
                except:
                    pass
        if count > 0:
            self._log(f"{Fore.GREEN}✅ Generated {count} .lab files.")
        else:
            self._log(f"{Fore.BLUE}ℹ️ .lab files already exist.")

    def extract_vqgan_tokens(self):
        """Step 1: Extract VQ Tokens using Official Config"""
        self._check_gpu_status()

        # 1. GET CORRECT CHECKPOINT
        checkpoint_path = self._ensure_checkpoint()

        # 2. CONFIGURATION FROM DOCS
        # Docs say: --config-name "modded_dac_vq"
        config_name = "modded_dac_vq"

        workers = "2" if os.name == 'nt' else "4"

        cmd = [
            sys.executable, str(self.extract_script),
            str(self.dataset_root),
            "--num-workers", workers,
            "--batch-size", "4",
            "--config-name", config_name,  # <--- CORRECT CONFIG
            "--checkpoint-path", str(checkpoint_path)  # <--- CORRECT CHECKPOINT
        ]

        self._run_subprocess_interactive(cmd, "VQGAN Token Extraction")

    def pack_dataset(self):
        """Step 2: Pack Dataset"""
        workers = "2" if os.name == 'nt' else "4"

        cmd = [
            sys.executable, str(self.build_script),
            "--input", str(self.dataset_root),
            "--output", str(self.proto_output),
            "--num-workers", workers,
            "--text-extension", ".lab"  # Explicitly state .lab
        ]

        self._run_subprocess_interactive(cmd, "Dataset Packing")

    def run(self):
        self.convert_txt_to_lab()
        self.extract_vqgan_tokens()
        self.pack_dataset()


if __name__ == "__main__":
    # --- DYNAMIC CONFIGURATION ---
    PROJECT_ROOT = Path(__file__).resolve().parent

    REPO_DIR = PROJECT_ROOT
    DATASET_DIR = PROJECT_ROOT / "dataset_final"
    OUTPUT_DIR = PROJECT_ROOT / "fish_training_data"

    # We point to the main checkpoints folder. The script handles the subfolder "openaudio-s1-mini"
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

    print(f"{Style.BRIGHT}--- FISH DATA PREP (Official v5.0) ---\n")
    print(f"📂 PROJECT ROOT: {PROJECT_ROOT}")
    print(f"📂 DATASET:      {DATASET_DIR}")
    print(f"💾 OUTPUT:       {OUTPUT_DIR}\n")

    try:
        builder = FishDataBuilder(DATASET_DIR, REPO_DIR, CHECKPOINTS_DIR, OUTPUT_DIR)
        builder.run()
    except Exception as e:
        print(f"\n{Fore.RED}🛑 Fatal Error: {e}")