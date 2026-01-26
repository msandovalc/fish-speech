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
        import torch
        import os
        import subprocess
        import sys
        from omegaconf.listconfig import ListConfig
        from omegaconf.dictconfig import DictConfig

        # Autorización de seguridad para PyTorch
        torch.serialization.add_safe_globals([ListConfig, DictConfig])
        torch.cuda.empty_cache()

        # 1. Matamos cualquier proceso previo para limpiar la 4090
        os.system("pkill -9 python")

        # DEBUG: Verificamos dónde estamos y persistencia
        print(f"🚀 Iniciando entrenamiento directo...")
        print(f"📍 CWD actual: {os.getcwd()}")
        print(f"📍 self.root: {str(self.root)}")
        print(f"📍 self.project_name: {self.project_name}")

        # Configuramos directorio de resultados EXPLÍCITO en /workspace (persistente)
        results_dir = "/workspace/fish-speech/results"
        os.makedirs(results_dir, exist_ok=True)
        project_path = os.path.join(results_dir, self.project_name)
        os.makedirs(project_path, exist_ok=True)

        print(f"📍 Resultados se guardarán en: {project_path}/checkpoints/")
        print(f"📍 Verificando persistencia...")
        print(f"📍 df -h: {os.popen('df -h /workspace').read()}")

        cmd = [
            sys.executable, str(self.train_script),
            "--config-name", "text2semantic_finetune",
            f"project={self.project_name}",
            f"train_dataset.proto_files=['{str(self.data_protos)}']",
            f"val_dataset.proto_files=['{str(self.data_protos)}']",
            f"pretrained_ckpt_path={str(self.base_model_path)}",

            # LoRA
            "+lora@model.model.lora_config=r_8_alpha_16",

            # Configuración de velocidad que ya vimos que funciona (1.14 it/s)
            "data.batch_size=1",
            "trainer.devices=1",
            "++trainer.accumulate_grad_batches=16",
            "++trainer.precision=bf16-mixed",
            "++trainer.max_steps=5000",

            # --- GUARDADO CORREGIDO (SINTAXIS HYDRA VÁLIDA) ---
            f"trainer.default_root_dir={project_path}",

            # Callback que SIEMPRE guarda (sin monitor problemático)
            "++callbacks.model_checkpoint.every_n_train_steps=250",
            "++callbacks.model_checkpoint.save_top_k=-1",
            "++callbacks.model_checkpoint.filename=epoch={epoch}-step={step}-loss={train_loss:.2f}",

            # Val check
            "++trainer.val_check_interval=250",

            # Logs también en el mismo directorio (SIN corchetes)
            f"logger[0].save_dir={project_path}",
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{str(self.root)}{os.pathsep}{env.get('PYTHONPATH', '')}"

        try:
            print("🔥 Ejecutando comando:", " ".join(cmd))
            subprocess.check_call(cmd, cwd=str(self.root), env=env)
            print("✅ Entrenamiento completado. Revisa:")
            print(f"   ls -la {project_path}/checkpoints/")
            os.system(f"ls -la {project_path}/checkpoints/")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    PROJECT_ROOT = Path("/workspace/fish-speech")
    PROJECT_NAME = "camila_voice_v1_stable"
    trainer = FishTrainer(PROJECT_ROOT, PROJECT_NAME)
    trainer.train()