import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger

# Optimización para GPUs de 4GB
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["EINX_FILTER_TRACEBACK"] = "false"

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio


class FishSpanishInference:
    def __init__(self, project_root: str = r"F:\Development\Pycharm\Projects\fish-speech"):
        self.device = "cuda"
        self.project_root = Path(project_root)
        self.checkpoint_dir = self.project_root / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half
        self.engine = self._load_models()

    def _load_models(self):
        logger.info("🛰️ Cargando motor Fish Speech (S1-Mini)...")
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.checkpoint_dir,
            device=self.device,
            precision=self.precision,
            compile=False
        )
        decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=self.checkpoint_dir / "codec.pth",
            device=self.device,
        )
        return TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=self.precision,
            compile=False
        )

    def generate(self, text: str, ref_audio_path: str, output_name: str = "clonacion_final_es.wav"):
        # 1. OPTIMIZACIÓN DE VRAM: Recortar audio de referencia
        # 16 segundos es demasiado para 4GB.
        # Cargamos y cortamos a los primeros 5 segundos.
        audio, sr = sf.read(ref_audio_path)
        if len(audio) > sr * 7:
            logger.info("✂️ Recortando audio de referencia a 5 segundos para ahorrar VRAM...")
            audio = audio[:sr * 5]

        # Guardar temporalmente el recorte para obtener los bytes
        tmp_ref = "tmp_ref.wav"
        sf.write(tmp_ref, audio, sr)
        with open(tmp_ref, "rb") as f:
            ref_audio_bytes = f.read()

        request = ServeTTSRequest(
            text=text,
            references=[ServeReferenceAudio(audio=ref_audio_bytes, text="")],
            max_new_tokens=512,
            chunk_length=150,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            format="wav"
        )

        logger.info("🎙️ Sintetizando...")
        # 2. FIX DEL TYPEERROR: Manejo de tuplas en el resultado
        results = list(self.engine.inference(request))

        audio_parts = []
        for res in results:
            # Si es una tupla, el audio suele ser el primer elemento
            if isinstance(res, tuple):
                audio_parts.append(res[0])
            # Si es un objeto con atributo audio
            elif hasattr(res, 'audio') and res.audio:
                audio_parts.append(res.audio)
            # Si ya son bytes
            elif isinstance(res, (bytes, bytearray)):
                audio_parts.append(res)

        if not audio_parts:
            logger.error("❌ No se obtuvieron datos de audio.")
            return

        audio_data = b"".join(audio_parts)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        output_path = str(Path(self.project_root) / output_name)
        sf.write(output_path, audio_np, 44100)
        logger.success(f"✅ ¡LOGRADO! Audio guardado en: {output_path}")

        # Limpieza
        if os.path.exists(tmp_ref): os.remove(tmp_ref)


if __name__ == "__main__":
    tts = FishSpanishInference()
    MI_TEXTO = "La causa de todo es mental, la mente es la que produce todo en la vida del individuo."
    RUTA_REF = r"F:\Development\Pycharm\Projects\fish-speech\voice_to_clone.wav"
    tts.generate(MI_TEXTO, RUTA_REF)