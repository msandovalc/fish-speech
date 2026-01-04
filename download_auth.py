import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import snapshot_download

# 1. FORZAR BARRAS DE PROGRESO
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
# Desactivar logs basura de HTTP para ver solo la barra
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

PROJECT_ROOT = Path(r"F:\Development\Pycharm\Projects\fish-speech")
sys.path.append(str(PROJECT_ROOT))

from tools.download_models import check_and_download_files as original_download


class AuthenticatedFishDownloader:

    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.repo_id = "fishaudio/openaudio-s1-mini"
        self.local_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"

    def download_all(self):
        """Descarga sincronizada con barra de progreso visual"""
        print(f"🛰️ Conectando a Hugging Face para descargar S1 Mini...")

        try:
            # snapshot_download gestiona model.pth y codec.pth con progreso nativo
            snapshot_download(
                repo_id=self.repo_id,
                local_dir=self.local_dir,
                token=self.hf_token,
                local_dir_use_symlinks=False,
                tqdm_class=tqdm,
                # Evita que se quede colgado en metadatos Xet
                revision="main"
            )
            print(f"\n✅ Pesos del modelo descargados en: {self.local_dir}")

        except Exception as e:
            print(f"\n❌ Error crítico en la descarga: {e}")

        # Descarga de ejecutables usando la lógica del proyecto
        print("\n🛠️ Verificando herramientas de sistema (FFmpeg/ASR)...")
        original_download("fishaudio/fish-speech-1", ["ffmpeg.exe", "ffprobe.exe"], str(PROJECT_ROOT))
        original_download("SpicyqSama007/fish-speech-packed", ["asr-label-win-x64.exe"], str(PROJECT_ROOT))


if __name__ == "__main__":
    # FIX: Ahora el código busca una variable de entorno llamada HF_TOKEN
    MI_TOKEN = os.getenv("HF_TOKEN")

    if not MI_TOKEN:
        print("❌ Error: No se encontró la variable de entorno HF_TOKEN")
    else:
        downloader = AuthenticatedFishDownloader(MI_TOKEN)
        downloader.download_all()