import os
import sys
import torch
import gc
import shutil
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger
from datetime import datetime

# --- CONFIGURACIÓN DE LOGS TRACE (PROTEGIDOS) ---
logger.remove()
logger.add(sys.stdout, colorize=True, level="TRACE",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
PROJECT_ROOT = Path("/kaggle/working/fish-speech")

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio


class FishFineTuningLab:
    def __init__(self):
        self.device = "cuda"
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half

        logger.info("🎯 INICIANDO SINTONÍA FINA: MODO FLUIDEZ Y AGUDEZA")
        self.engine = self._load_models()

    def _load_models(self):
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.checkpoint_dir,
            device=self.device, precision=self.precision, compile=True
        )
        decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=self.checkpoint_dir / "codec.pth",
            device=self.device
        )
        return TTSInferenceEngine(llama_queue=llama_queue, decoder_model=decoder_model, precision=self.precision,
                                  compile=False)

    def run_fine_tuning(self, text, prompt_text, ref_path, num_tests=20):
        # PASO 1: VQ TOKENS (Fijos)
        logger.trace(f"🧬 [PASO 1] Codificando referencia ganadora: {ref_path}")
        with open(ref_path, "rb") as f:
            audio_bytes = f.read()

        with torch.inference_mode():
            vq_tokens = self.engine.encode_reference(audio_bytes, enable_reference_audio=True)
            logger.debug("✅ ADN Vocal cargado para refinamiento.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = PROJECT_ROOT / f"sintonia_fina_{timestamp}"
        folder.mkdir(parents=True, exist_ok=True)

        # RANGOS AJUSTADOS (Buscando el pico de fluidez y tono agudo)
        t_start, t_end = 0.74, 0.82  # Centrado en tus favoritos
        p_start, p_end = 0.81, 0.86  # Subimos Top_P para más brillo/agudos

        for i in range(num_tests):
            progress = i / (num_tests - 1) if num_tests > 1 else 0
            curr_t = round(t_start + (t_end - t_start) * progress, 2)
            curr_p = round(p_start + (p_end - p_start) * progress, 2)

            # Penalización suave para reducir lo "robótico" sin perder claridad
            curr_pen = round(1.15 + (0.1 * progress), 2)

            # CHUNK AUMENTADO: 350-450 para mejorar la fluidez entre palabras
            curr_chunk = int(350 + (100 * (1 - progress)))

            logger.trace("-" * 50)
            logger.trace(f"🌀 [TEST {i + 1}/{num_tests}] | OBJETIVO: Fluidez Aguda")
            logger.trace(f"   ∟ PARÁMETROS: T={curr_t} | P={curr_p} | Pen={curr_pen} | Chunk={curr_chunk}")

            name = f"FINA_{i + 1:02d}_T{curr_t}_P{curr_p}_C{curr_chunk}"

            req = ServeTTSRequest(
                text=text,
                references=[ServeReferenceAudio(
                    audio=audio_bytes,
                    tokens=vq_tokens.tolist(),
                    text=prompt_text
                )],
                max_new_tokens=1024,
                chunk_length=curr_chunk,
                top_p=curr_p,
                temperature=curr_t,
                repetition_penalty=curr_pen,
                format="wav"
            )

            try:
                results = self.engine.inference(req)
                audio_parts = []
                for res in results:
                    chunk_data = res.audio if hasattr(res, 'audio') else res
                    if isinstance(chunk_data, tuple):
                        for item in chunk_data:
                            if isinstance(item, np.ndarray): audio_parts.append(item)
                    elif isinstance(chunk_data, np.ndarray):
                        audio_parts.append(chunk_data)

                if audio_parts:
                    final_audio = np.concatenate(audio_parts)
                    sf.write(str(folder / f"{name}.wav"), final_audio, 44100)
                    logger.debug(f"💾 Guardado: {name}.wav | VRAM: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB libres")

                if (i + 1) % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                logger.error(f"❌ Fallo en test {i + 1}: {e}")

        zip_path = PROJECT_ROOT / f"clon_final_paunel_{timestamp}"
        shutil.make_archive(str(zip_path), 'zip', folder)
        logger.success(f"🏁 SINTONÍA FINA COMPLETADA. ZIP: {zip_path}.zip")


if __name__ == "__main__":
    lab = FishFineTuningLab()

    # TEXTO PARA PAUNEL: Mezcla de afirmación y puntuación para tono agudo
    TEXTO_PAUNEL = "¡La mente es la causa de todo! ¿Produce la realidad del individuo con total claridad? ¡Claro que sí!"

    PROMPT_TEXT = ("Agradezco que cada vez trabajo menos y gano más, estoy tan feliz y agradecida "
                   "ahora que el dinero viene a mi en cantidades cada vez mayores de diversar "
                   "fuentes de forma continua y correcta, soy abundante.")

    lab.run_fine_tuning(TEXTO_PAUNEL, PROMPT_TEXT, "/kaggle/working/fish-speech/voice_to_clone.wav")