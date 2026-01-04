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

# --- CONFIGURACIÓN DE LOGS TRACE (MÁXIMA VISIBILIDAD) ---
logger.remove()
logger.add(sys.stdout, colorize=True, level="TRACE",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
PROJECT_ROOT = Path("/kaggle/working/fish-speech")

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio


class FishRefinementLab:
    def __init__(self):
        self.device = "cuda"
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half

        logger.info("🚀 INICIANDO REFINAMIENTO PARA TONOS AGUDOS (PAUNEL)")
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

    def run_refinement_grid(self, text, prompt_text, ref_path, num_tests=30):
        # --- PASO 1 OFICIAL: Extracción de tokens VQ ---
        logger.trace(f"🧬 [PASO 1] Codificando referencia para refinamiento: {ref_path}")
        with open(ref_path, "rb") as f:
            audio_bytes = f.read()

        with torch.inference_mode():
            vq_tokens = self.engine.encode_reference(audio_bytes, enable_reference_audio=True)
            logger.debug(f"✅ ADN Vocal listo. Inyectando tokens en Paso 2/3.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = PROJECT_ROOT / f"refinamiento_agudo_{timestamp}"
        folder.mkdir(parents=True, exist_ok=True)

        # RANGOS AJUSTADOS (Basados en V46 y V09 para buscar más agudeza)
        # Empezamos desde donde estaban V46/V09 y subimos
        t_start, t_end = 0.65, 0.95
        p_start, p_end = 0.75, 0.90

        for i in range(num_tests):
            progress = i / (num_tests - 1) if num_tests > 1 else 0
            curr_t = round(t_start + (t_end - t_start) * progress, 2)
            curr_p = round(p_start + (p_end - p_start) * progress, 2)
            curr_pen = round(1.2 + (0.15 * progress), 2)  # Penalización ligeramente alta para evitar ronquera
            curr_chunk = 400 if i % 2 == 0 else 250  # Chunks cortos (250) suelen forzar tonos más altos

            logger.trace("-" * 50)
            logger.trace(f"🌀 [REFINAMIENTO {i + 1}/{num_tests}] | AGUDEZA PREVISTA: {progress:.1%}")
            logger.trace(f"   ∟ PARÁMETROS: Temp={curr_t} | Top_P={curr_p} | Penalty={curr_pen} | Chunk={curr_chunk}")

            name = f"AGUDO_{i + 1:02d}_T{curr_t}_P{curr_p}_C{curr_chunk}"

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
                    logger.debug(f"💾 Guardado con éxito: {name}.wav")

                if (i + 1) % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                logger.error(f"❌ Error en ciclo {i + 1}: {e}")

        zip_path = PROJECT_ROOT / f"resultados_agudos_paunel_{timestamp}"
        shutil.make_archive(str(zip_path), 'zip', folder)
        logger.success(f"🏁 REFINAMIENTO COMPLETADO. ZIP generado: {zip_path}.zip")


if __name__ == "__main__":
    lab = FishRefinementLab()

    # TEXTO CON "ENFASIS": Añadimos signos para forzar un tono más alto/agudo
    TEXTO_PRUEBA = "¡La mente es la causa de todo! ¿Produce la realidad del individuo con total claridad? ¡Cree en ello!"

    PROMPT_TEXT = ("Agradezco que cada vez trabajo menos y gano más, estoy tan feliz y agradecida "
                   "ahora que el dinero viene a mi en cantidades cada vez mayores de diversar "
                   "fuentes de forma continua y correcta, soy abundante.")

    lab.run_refinement_grid(TEXTO_PRUEBA, PROMPT_TEXT, "/kaggle/working/fish-speech/voice_to_clone.wav", num_tests=30)