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

# --- CONFIGURACIÓN DE LOGS TRACE (PROHIBIDO QUITAR) ---
logger.remove()
logger.add(sys.stdout, colorize=True, level="TRACE",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
PROJECT_ROOT = Path("/kaggle/working/fish-speech")

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

# --- LISTA GLOBAL DE PRESETS GANADORES ---
VOICE_PRESETS = {
    "MARLENE": {
        "temp": 0.78,
        "top_p": 0.89,
        "chunk": 517,
        "penalty": 1.12,
        "ref_path": "/kaggle/working/fish-speech/ElevenLabs_2026-01-04T18_49_10_Marlene.mp3",
        "prompt": "La mente lo es todo. La causa mental. La causa de todo -absolutamente todo- es mental, es decir, la mente es la que produce o causa todo en la vida del individuo. Cuando reconozcamos, entendamos y aceptemos esta verdad, habremos dado un paso muy importante en el progreso del desarrollo. Si todo es mental, este es un universo mental, donde todo funciona por medios mentales. Nosotros somos seres mentales, mentalidades buenas, perfectas y eternas. La mente sólo tiene una actividad, pensar. El pensamiento es todo lo de la mente lo único que somos y tenemos es pensamiento, por ello, el pensamiento es lo más importante de todo."
    },
    "ALEJANDRO": {
        "temp": 0.81,
        "top_p": 0.90,
        "chunk": 607,
        "penalty": 1.12,
        "ref_path": "/kaggle/working/fish-speech/ElevenLabs_2026-01-04T19_56_14_Alejandro.mp3",
        "prompt": "La mente lo es todo. La causa mental. La causa de todo -absolutamente todo- es mental, es decir, la mente es la que produce o causa todo en la vida del individuo. Cuando reconozcamos, entendamos y aceptemos esta verdad, habremos dado un paso muy importante en el progreso del desarrollo. Si todo es mental, este es un universo mental, donde todo funciona por medios mentales. Nosotros somos seres mentales, mentalidades buenas, perfectas y eternas. La mente sólo tiene una actividad, pensar. El pensamiento es todo lo de la mente lo único que somos y tenemos es pensamiento, por ello, el pensamiento es lo más importante de todo."
    }
}


class FishProductionLab:
    def __init__(self):
        self.device = "cuda"
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half
        logger.info("🎯 MODO PRODUCCIÓN Y EXPERIMENTACIÓN MULTIVOZ")
        self.engine = self._load_models()

    def _load_models(self):
        llama_queue = launch_thread_safe_queue(checkpoint_path=self.checkpoint_dir, device=self.device,
                                               precision=self.precision, compile=True)
        decoder_model = load_decoder_model(config_name="modded_dac_vq",
                                           checkpoint_path=self.checkpoint_dir / "codec.pth", device=self.device)
        return TTSInferenceEngine(llama_queue=llama_queue, decoder_model=decoder_model, precision=self.precision,
                                  compile=False)

    def run_hyper_search(self, text, num_tests=15):
        """
        Realiza una búsqueda incremental para TODAS las voces en VOICE_PRESETS.
        Ancla los rangos a los valores base de cada voz para buscar fluidez y agudeza.
        """
        logger.info(f"🧪 Iniciando Laboratorio Hyper-Search para {len(VOICE_PRESETS)} voces.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for voice_name, base_params in VOICE_PRESETS.items():
            logger.trace("=" * 60)
            logger.info(f"🎤 PROCESANDO PRUEBAS PARA: {voice_name}")

            # Crear carpeta específica para la voz
            voice_folder = PROJECT_ROOT / f"hyper_{voice_name}_{timestamp}"
            voice_folder.mkdir(parents=True, exist_ok=True)

            # Paso 1: Codificación de la referencia específica de la voz
            logger.trace(f"🧬 [PASO 1] Codificando ADN Vocal de {voice_name}...")
            with open(base_params['ref_path'], "rb") as f:
                audio_bytes = f.read()
            with torch.inference_mode():
                vq_tokens = self.engine.encode_reference(audio_bytes, enable_reference_audio=True)

            # Definición de rangos anclados a la base de la voz
            t_base = base_params['temp']
            p_base = base_params['top_p']

            for i in range(num_tests):
                progress = i / (num_tests - 1) if num_tests > 1 else 0

                # Evolución: de la base hacia arriba para más expresión y agudeza
                curr_t = round(t_base + (0.10 * progress), 2)
                # Capamos Top_P en 0.95 para evitar ruidos extraños
                curr_p = round(min(p_base + (0.05 * progress), 0.95), 2)
                # Forzamos fluidez con chunks de 700 a 1000
                curr_chunk = int(700 + (300 * progress))
                # Penalización baja para favorecer la fluidez (conexión de palabras)
                curr_pen = round(1.05 + (0.10 * progress), 2)

                logger.trace(f"🌀 [{voice_name} {i + 1}/{num_tests}] | T={curr_t} | P={curr_p} | C={curr_chunk}")

                name = f"{voice_name}_FINA_{i + 1:02d}_T{curr_t}_P{curr_p}_C{curr_chunk}"

                req = ServeTTSRequest(
                    text=text,
                    references=[
                        ServeReferenceAudio(audio=audio_bytes, tokens=vq_tokens.tolist(), text=base_params['prompt'])],
                    max_new_tokens=1024,
                    chunk_length=curr_chunk,
                    top_p=curr_p,
                    temperature=curr_t,
                    repetition_penalty=curr_pen,
                    format="wav"
                )

                try:
                    results = self.engine.inference(req)
                    audio_parts = [res.audio if hasattr(res, 'audio') else res for res in results]
                    clean_parts = []
                    for p in audio_parts:
                        if isinstance(p, tuple):
                            clean_parts.extend([x for x in p if isinstance(x, np.ndarray)])
                        elif isinstance(p, np.ndarray):
                            clean_parts.append(p)

                    if clean_parts:
                        sf.write(str(voice_folder / f"{name}.wav"), np.concatenate(clean_parts), 44100)
                        logger.debug(f"💾 Guardado: {name}.wav")

                    if (i + 1) % 5 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

                except Exception as e:
                    logger.error(f"❌ Error en {voice_name} Ciclo {i + 1}: {e}")

            # Generar ZIP por cada voz al terminar su batería
            zip_name = f"pruebas_{voice_name}_{timestamp}"
            shutil.make_archive(str(PROJECT_ROOT / zip_name), 'zip', voice_folder)
            logger.success(f"🏁 BATERÍA {voice_name} TERMINADA -> {zip_name}.zip")

    def generate_production_batch(self, text_to_speak):
        # ... (Se mantiene igual para tus audios finales con presets) ...
        pass


if __name__ == "__main__":
    lab = FishProductionLab()

    TEXTO_TEST = """
    El gran secreto para llegar al entendimiento de la verdad es mantener nuestro pensamiento en el bien,  
    en forma continua. Esto causará, invariablemente, todo lo bueno en la vida del individuo.
    """

    # Ahora la función Hyper Search procesa automáticamente a Marlene y Alejandro
    # usando sus propios presets como punto de partida.
    lab.run_hyper_search(TEXTO_TEST, num_tests=15)