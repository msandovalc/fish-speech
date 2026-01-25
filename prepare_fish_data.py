import os
import logging
import subprocess
import sys
import shutil
import torch
from pathlib import Path
from colorama import init, Fore, Style
from huggingface_hub import hf_hub_download

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

        self.extract_script = self.fish_root / "tools" / "vqgan" / "extract_vq.py"
        self.build_script = self.fish_root / "tools" / "llama" / "build_dataset.py"

        self.proto_output = self.output_dir / "protos"
        self.proto_output.mkdir(parents=True, exist_ok=True)

        self._log(f"{Fore.CYAN}🚀 Initializing Fish Data Builder (v6.0 RunPod Fix)...")
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

    def _check_gpu_status(self):
        self._log(f"{Fore.YELLOW}🔍 Checking Hardware Acceleration...")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            self._log(f"{Fore.GREEN}   ✅ CUDA DETECTED: {gpu_name} ({vram:.1f} GB VRAM)")
            # Liberar basura previa
            torch.cuda.empty_cache()
        else:
            self._log(f"{Fore.RED}   ⚠️ NO GPU DETECTED.")

    def _ensure_checkpoint(self):
        """Usa el checkpoint que ya descargamos en la carpeta de openaudio-s1-mini"""
        filename = "codec.pth"
        local_model_path = self.checkpoints_dir / "openaudio-s1-mini" / filename

        if not local_model_path.exists():
            self._log(f"{Fore.YELLOW}⚠️ Checkpoint no encontrado en {local_model_path}. Descargando...")
            local_model_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_path = hf_hub_download(
                repo_id="fishaudio/openaudio-s1-mini",
                filename=filename,
                local_dir=self.checkpoints_dir / "openaudio-s1-mini"
            )
            return Path(downloaded_path)

        self._log(f"{Fore.GREEN}   ✅ Checkpoint found: {local_model_path}")
        return local_model_path

    def _run_subprocess_interactive(self, cmd, description):
        self._log(f"{Fore.MAGENTA}⚙️  Running: {description}...")
        env = os.environ.copy()

        # FIX PARA MEMORIA EN RUNPOD
        env["PYTHONPATH"] = f"{str(self.fish_root)}{os.pathsep}{env.get('PYTHONPATH', '')}"
        env["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Evita fragmentación

        try:
            subprocess.check_call(cmd, cwd=str(self.fish_root), env=env)
            self._log(f"{Fore.GREEN}✅ {description} finished successfully.")
        except subprocess.CalledProcessError as e:
            self._log(f"{Fore.RED}❌ {description} Failed (Exit code {e.returncode}).", level="error")
            sys.exit(1)

    def convert_txt_to_lab(self):
        self._log(f"{Fore.YELLOW}🔄 Normalizing extensions (.txt -> .lab)...")
        count = 0
        for txt_file in self.dataset_root.rglob("*.txt"):
            lab_file = txt_file.with_suffix(".lab")
            if not lab_file.exists():
                shutil.copy2(txt_file, lab_file)
                count += 1
        self._log(f"{Fore.BLUE}ℹ️ Normalización terminada.")

    def extract_vqgan_tokens(self):
        self._check_gpu_status()
        checkpoint_path = self._ensure_checkpoint()

        # CAMBIO CLAVE: workers=1 y batch=8 para evitar OOM
        cmd = [
            sys.executable, str(self.extract_script),
            str(self.dataset_root),
            "--num-workers", "1",
            "--batch-size", "8",
            "--config-name", "modded_dac_vq",
            "--checkpoint-path", str(checkpoint_path)
        ]
        self._run_subprocess_interactive(cmd, "VQGAN Token Extraction")

    def pack_dataset(self):
        cmd = [
            sys.executable, str(self.build_script),
            "--input", str(self.dataset_root),
            "--output", str(self.proto_output),
            "--num-workers", "1",
            "--text-extension", ".lab"
        ]
        self._run_subprocess_interactive(cmd, "Dataset Packing")

    def run(self):
        self.convert_txt_to_lab()
        self.extract_vqgan_tokens()
        self.pack_dataset()


if __name__ == "__main__":
    PROJECT_ROOT = Path("/workspace/fish-speech")
    REPO_DIR = PROJECT_ROOT
    DATASET_DIR = PROJECT_ROOT / "dataset_final"
    OUTPUT_DIR = PROJECT_ROOT / "fish_training_data"
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

    print(f"{Style.BRIGHT}--- FISH DATA PREP (RunPod v6.0) ---\n")

    try:
        builder = FishDataBuilder(DATASET_DIR, REPO_DIR, CHECKPOINTS_DIR, OUTPUT_DIR)
        builder.run()
    except Exception as e:
        print(f"\n{Fore.RED}🛑 Fatal Error: {e}")