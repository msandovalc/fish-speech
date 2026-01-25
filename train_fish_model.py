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

        # Paths oficiales en RunPod
        self.checkpoints_dir = self.root / "checkpoints"
        self.data_protos = self.root / "fish_training_data" / "protos"
        self.train_script = self.root / "fish_speech" / "train.py"

        # Ruta al modelo que bajamos (openaudio-s1-mini)
        self.base_model_path = base_model_path or (self.checkpoints_dir / "openaudio-s1-mini")

        print(f"{Fore.CYAN}🚀 Initializing RunPod Trainer (RTX 4090 Optimized)")
        print(f"   🧠 Base Model Path: {self.base_model_path}")

        self._validate_paths()

    def _validate_paths(self):
        if not self.data_protos.exists():
            print(f"{Fore.RED}❌ Datos de entrenamiento no encontrados en: {self.data_protos}")
            sys.exit(1)

        # Buscamos model.pth que es el que bajamos de HF
        if not (self.base_model_path / "model.pth").exists():
            print(f"{Fore.RED}❌ No se encuentra 'model.pth' en {self.base_model_path}")
            sys.exit(1)

        print(f"{Fore.GREEN}   ✅ Estructura de archivos validada.")

    def train(self):
        torch.cuda.empty_cache()

        print(f"{Fore.MAGENTA}🔥 Iniciando LoRA Fine-Tuning en RTX 4090...")

        # Configuramos el comando optimizado para la 4090
        cmd = [
            sys.executable, str(self.train_script),
            "--config-name", "text2semantic_finetune",
            f"project={self.project_name}",

            # --- DATASET ---
            f"train_dataset.proto_files=['{str(self.data_protos)}']",
            f"val_dataset.proto_files=['{str(self.data_protos)}']",

            # --- MODELO ---
            f"pretrained_ckpt_path={str(self.base_model_path / 'model.pth')}",
            f"trainer.default_root_dir={self.root}/results/{self.project_name}",

            # --- LORA CONFIG ---
            "+lora@model.model.lora_config=r_8_alpha_16",

            # --- CONFIGURACIÓN DE PODER (RTX 4090) ---
            "data.batch_size=4",  # Subimos de 1 a 4 (VRAM de sobra)
            "trainer.devices=1",
            "trainer.accumulate_grad_batches=4",  # Batch efectivo = 16

            # BF16 es MUCHO mejor para la 4090 que FP16
            "trainer.precision=bf16-mixed",

            "data.num_workers=4",  # Carga de datos más rápida
            "trainer.max_epochs=20",  # Un poco más de épocas para mejor calidad
            "trainer.val_check_interval=50",  # Revisar progreso más seguido
        ]

        # Environment Fix
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{str(self.root)}{os.pathsep}{env.get('PYTHONPATH', '')}"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        try:
            subprocess.check_call(cmd, cwd=str(self.root), env=env)
            print(f"\n{Fore.GREEN}✨ ¡ENTRENAMIENTO COMPLETADO CON ÉXITO!")
            print(f"   💾 Tus checkpoints están en: {self.root}/results/{self.project_name}")

        except Exception as e:
            print(f"\n{Fore.RED}❌ Error en el entrenamiento: {e}")


if __name__ == "__main__":
    # Ajuste manual para asegurar que apunte a /workspace
    PROJECT_ROOT = Path("/workspace/fish-speech")
    PROJECT_NAME = "camila_voice_v1"

    trainer = FishTrainer(PROJECT_ROOT, PROJECT_NAME)
    trainer.train()