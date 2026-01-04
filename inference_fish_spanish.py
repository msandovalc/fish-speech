import os
import sys
import torch
import shutil
import random
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger
from datetime import datetime
from scipy.spatial.distance import cosine

# --- MANTENEMOS Y AUMENTAMOS LOGS TRACE ---
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    level="TRACE",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
)

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["EINX_FILTER_TRACEBACK"] = "false"

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio


class FishAuditLab:
    def __init__(self):
        self.device = "cuda"
        self.checkpoint_dir = PROJECT_ROOT / "checkpoints" / "openaudio-s1-mini"
        self.precision = torch.half

        logger.info("🚀 INICIANDO LABORATORIO AUDITADO EN TESLA T4...")
        self.engine = self._load_models()

        # Cargamos el Auditor de Identidad (ECAPA-TDNN)
        from speechbrain.inference.speaker import EncoderClassifier
        logger.info("🕵️ Cargando Biometría de Voz (SpeechBrain)...")
        self.auditor = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-tdnn",
            run_opts={"device": self.device}
        )

    def _load_models(self):
        llama_queue = launch_thread_safe_queue(checkpoint_path=self.checkpoint_dir, device=self.device,
                                               precision=self.precision, compile=True)
        decoder_model = load_decoder_model(config_name="modded_dac_vq",
                                           checkpoint_path=self.checkpoint_dir / "codec.pth", device=self.device)
        return TTSInferenceEngine(llama_queue=llama_queue, decoder_model=decoder_model, precision=self.precision,
                                  compile=True)

    def get_voice_dna(self, audio_path):
        """Extrae la identidad vocal para comparación."""
        signal, fs = librosa.load(audio_path, sr=16000)
        with torch.no_grad():
            emb = self.auditor.encode_batch(torch.tensor(signal).unsqueeze(0).to(self.device))
        return emb.squeeze().cpu().numpy()

    def run_full_audit(self, text, ref_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = PROJECT_ROOT / f"mega_test_{timestamp}"
        folder.mkdir(parents=True, exist_ok=True)

        with open(ref_path, "rb") as f:
            ref_bytes = f.read()

        logger.trace(f"🧬 Generando ADN de referencia para: {Path(ref_path).name}")
        ref_dna = self.get_voice_dna(ref_path)

        # 1. BATERÍA DE 50 PRUEBAS
        results_data = []
        logger.info(f"🧪 Iniciando 50 variantes. Rango de Temp corregido (0.1 - 1.0)")

        for i in range(1, 51):
            # CORRECCIÓN DE ERROR: Temperatura máx 1.0
            t = round(random.uniform(0.3, 1.0), 2)
            p = round(random.uniform(0.6, 0.9), 2)
            penalty = round(random.uniform(1.1, 1.5), 2)
            chunk = random.choice([250, 350, 500])

            name = f"V{i:02d}_T{t}_P{p}_Pen{penalty}_C{chunk}"
            logger.trace(f"🌀 [Variante {i}/50] Generando con T={t}, P={p}, Pen={penalty}")

            req = ServeTTSRequest(
                text=text,
                references=[ServeReferenceAudio(audio=ref_bytes, text="")],
                max_new_tokens=1024,
                chunk_length=chunk,
                top_p=p,
                temperature=t,
                repetition_penalty=penalty,
                format="wav"
            )

            results = self.engine.inference(req)
            chunks = []
            sr = 44100
            for res in results:
                item = res.audio if hasattr(res, 'audio') else res
                if isinstance(item, tuple):
                    for sub in item:
                        if isinstance(sub, int):
                            sr = sub
                        elif isinstance(sub, np.ndarray):
                            chunks.append(sub)
                elif isinstance(item, np.ndarray):
                    chunks.append(item)

            if chunks:
                audio_np = np.concatenate(chunks)
                path = folder / f"{name}.wav"
                sf.write(str(path), audio_np, sr)

                # 2. AUDITORÍA INMEDIATA
                gen_dna = self.get_voice_dna(str(path))
                similarity = 1 - cosine(ref_dna, gen_dna)
                results_data.append({"name": name, "sim": similarity, "path": path})

                logger.debug(f"📊 [Chunk {i}] Match de Identidad: {similarity:.2%}")
            else:
                logger.error(f"❌ Falló variante {i}")

        # 3. RESUMEN Y EMPAQUETADO
        results_data.sort(key=lambda x: x['sim'], reverse=True)

        logger.info("--- 🏆 TOP 5 VARIANTES QUE MÁS SE PARECEN A TU VOZ ---")
        for idx, res in enumerate(results_data[:5]):
            logger.success(f"{idx + 1}. {res['name']} -> MATCH: {res['sim']:.2%}")

        zip_path = PROJECT_ROOT / f"audit_results_{timestamp}"
        shutil.make_archive(str(zip_path), 'zip', folder)
        logger.success(f"🏁 Todo listo. ZIP generado: {zip_path}.zip")


if __name__ == "__main__":
    lab = FishAuditLab()
    TEXTO = "La mente es la causa de todo; produce la realidad del individuo, ¡con total claridad!"
    REFERENCIA = "/kaggle/working/fish-speech/voice_to_clone.wav"
    lab.run_full_audit(TEXTO, REFERENCIA)