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
    # --- FIX PARA PYTORCH 2.6 (Seguridad de carga) ---
    import torch
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    torch.serialization.add_safe_globals([ListConfig, DictConfig])
    # --------------------------------------------------

    torch.cuda.empty_cache()

    # Definimos la ruta absoluta para los checkpoints
    abs_ckpt_dir = f"{self.root}/results/{self.project_name}/checkpoints"

    print(f"{Fore.MAGENTA}🔥 Starting Stable LoRA (Batch 2 - RTX 4090)...")
    print(f"{Fore.MAGENTA}🔥 Configuración de Experto: Objetivo 5000 Pasos...")
    print(f"{Fore.CYAN}📍 Checkpoints se guardarán en: {abs_ckpt_dir}")

    cmd = [
        sys.executable, str(self.train_script),
        "--config-name", "text2semantic_finetune",
        f"project={self.project_name}",
        f"train_dataset.proto_files=['{str(self.data_protos)}']",
        f"val_dataset.proto_files=['{str(self.data_protos)}']",
        f"pretrained_ckpt_path={str(self.base_model_path)}",
        f"trainer.default_root_dir={self.root}/results/{self.project_name}",

        # --- LORA ---
        "+lora@model.model.lora_config=r_8_alpha_16",

        # --- AJUSTES DE PODER ---
        "data.batch_size=2",
        "trainer.devices=1",
        "++trainer.accumulate_grad_batches=8",
        "++trainer.precision=bf16-mixed",

        # --- CONTROL DE TIEMPO ---
        "++trainer.max_steps=5000",
        "++trainer.limit_train_batches=500",
        "++trainer.max_epochs=-1",

        # --- FIX CRÍTICO DE GUARDADO (Checkpoints cada 250 pasos) ---
        f"++callbacks.model_checkpoint.dirpath={abs_ckpt_dir}",
        "++callbacks.model_checkpoint.every_n_train_steps=250",
        "++callbacks.model_checkpoint.monitor=train/loss",
        "++callbacks.model_checkpoint.mode=min",
        "++callbacks.model_checkpoint.save_top_k=5",
        "++callbacks.model_checkpoint.auto_insert_metric_name=False",

        # Validación
        "++trainer.val_check_interval=250",
        "++trainer.limit_val_batches=1",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{str(self.root)}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        subprocess.check_call(cmd, cwd=str(self.root), env=env)
        print(f"\n{Fore.GREEN}✨ ENTRENAMIENTO EXITOSO!")
    except Exception as e:
        print(f"\n{Fore.RED}❌ El proceso se detuvo o falló. Revisa los logs arriba.")
        

if __name__ == "__main__":
    PROJECT_ROOT = Path("/workspace/fish-speech")
    PROJECT_NAME = "camila_voice_v1_stable"
    trainer = FishTrainer(PROJECT_ROOT, PROJECT_NAME)
    trainer.train()