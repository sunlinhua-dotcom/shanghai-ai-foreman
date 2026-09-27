import logging
import os

from .cosyvoice import CosyVoiceTTS  # noqa: F401
from .edge_tts import EdgeTTS  # noqa: F401
from .route import CachedTTS, TimeRoutedTTS  # noqa: F401
from .volc import FallbackTTS, VolcTTS  # noqa: F401
from .voxtts import VoxTTS  # noqa: F401

log = logging.getLogger("xz.tts")


def create_tts(cfg):
    """按 XZ_TTS_PROVIDER 选引擎：edge（默认，微软 Edge 神经语音，免 key）/ voxtts（VoxCPM2）
    / volc（豆包语音 2.0，出错退回 Edge 茉莉）。"""
    if cfg.tts_provider == "voxtts":
        return VoxTTS(
            base_url=cfg.tts_base_url,
            voice=cfg.tts_voice or "my_voice",
            timeout=cfg.tts_timeout,
        )
    if cfg.tts_provider == "volc":
        backup = EdgeTTS(sample_rate=cfg.downlink_sample_rate, timeout=min(cfg.tts_timeout, 30.0))
        if not cfg.volc_api_key:
            log.warning("XZ_TTS_PROVIDER=volc 但 .env 里没有 XZ_VOLC_API_KEY（敲 doubao-key 录），先用 Edge")
            return backup
        volc = VolcTTS(
            api_key=cfg.volc_api_key,
            voice=cfg.tts_voice,
            resource_id=cfg.volc_resource_id,
            instruction=cfg.volc_instruction,
            speech_rate=cfg.volc_speech_rate,
            sample_rate=cfg.downlink_sample_rate,
            timeout=min(cfg.tts_timeout, 20.0),
        )
        if cfg.tts_cache_dir:
            volc = CachedTTS(volc, os.path.join(cfg.tts_cache_dir, "volc"))
        if not cfg.cosyvoice_url:
            return FallbackTTS(volc, backup)
        # 9/26 起：豆包 → GX10 CosyVoice（本人声音讲上海话）→ Edge；23:00–11:00 直接从 CosyVoice 开始
        local = FallbackTTS(
            CosyVoiceTTS(cfg.cosyvoice_url, sample_rate=cfg.downlink_sample_rate, timeout=min(cfg.tts_timeout, 30.0)),
            backup,
        )
        return TimeRoutedTTS(FallbackTTS(volc, local), local, cfg.tts_local_hours, cfg.announce_tz)
    if cfg.tts_provider == "cosyvoice":
        # 只用自建的吴语配音（tools/wu_server.py），这句没出声就退 Edge
        return FallbackTTS(
            CosyVoiceTTS(cfg.cosyvoice_url, sample_rate=cfg.downlink_sample_rate, timeout=min(cfg.tts_timeout, 30.0)),
            EdgeTTS(sample_rate=cfg.downlink_sample_rate, timeout=min(cfg.tts_timeout, 30.0)),
        )
    return EdgeTTS(
        voice=cfg.tts_voice,
        sample_rate=cfg.downlink_sample_rate,
        timeout=min(cfg.tts_timeout, 30.0),  # 云端神经语音不该等 120s
    )
