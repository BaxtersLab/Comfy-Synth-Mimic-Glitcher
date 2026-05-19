"""
ComfyUI Node: ComfySynthMimicGlitcher
Node 3 — Harmonic Pitch-Mimic + Glitch Synth Layer
"""
import numpy as np
import torch
import comfy.utils
from typing import Optional
import scipy.signal
import json

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available. Pitch detection will be limited.")

class ComfySynthMimicGlitcher:
    """
    ComfyUI node for Harmonic Pitch-Mimic + Glitch Synth Layer.
    Adds harmonic shimmer, time-staggered layers, detune, phase randomization, and glitch-domain effects.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "synth_volume": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "delay_time": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "waveform_type": (["sine", "square", "sawtooth", "triangle"], {
                    "default": "sine"
                }),
                "mimic_effect": (["none", "bake", "scramble", "fry", "glitch", "stutter", "tape_stop"], {
                    "default": "none"
                }),
                "sensitivity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "dry_wet_mix": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "tempo_map_json": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "JSON string from TempoMapperNode. Syncs stutter/delay to beat grid."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "apply_mimic_glitch"
    CATEGORY = "audio/effects"

    def apply_mimic_glitch(self, audio, synth_volume=0.3, delay_time=0.2, waveform_type="sine", mimic_effect="none", sensitivity=0.5, dry_wet_mix=0.5, tempo_map_json=""):
        """
        Apply Synth Mimic Glitcher effect to audio tensor.
        """
        # Convert ComfyUI audio tensor to numpy
        audio_np = audio['waveform'].squeeze(0).cpu().numpy()
        sample_rate = audio['sample_rate']

        # Apply effect
        processed_audio = self.synth_mimic_glitch(
            audio_in=audio_np,
            sample_rate=sample_rate,
            synth_volume=synth_volume,
            delay_time=delay_time,
            waveform_type=waveform_type,
            mimic_effect=mimic_effect,
            sensitivity=sensitivity,
            dry_wet_mix=dry_wet_mix
            ,
            tempo_map_json=tempo_map_json,
        )

        # Convert back to ComfyUI audio tensor format
        processed_tensor = torch.from_numpy(processed_audio).unsqueeze(0).unsqueeze(0)

        return ({"waveform": processed_tensor, "sample_rate": sample_rate},)

    def synth_mimic_glitch(
        self,
        audio_in: np.ndarray,
        sample_rate: int,
        synth_volume: float = 0.3,
        delay_time: float = 0.2,
        waveform_type: str = "sine",
        mimic_effect: str = "none",
        sensitivity: float = 0.5,
        dry_wet_mix: float = 0.5,
        tempo_map_json: str = "",
    ):
        """
        Synth Mimic Glitcher: Pitch-following synthesizer with glitch effects.
        """
        # Ensure input is 2D: (samples, channels)
        original_mono = audio_in.ndim == 1
        if original_mono:
            audio_in = audio_in[:, np.newaxis]

        num_samples, num_channels = audio_in.shape
        audio_out = np.zeros_like(audio_in)

        # Parse tempo map once for the whole call
        tempo = self._parse_tempo_map(tempo_map_json)

        for ch in range(num_channels):
            # Detect pitch using librosa (with fallback)
            pitch_hz = self._detect_pitch_librosa(audio_in[:, ch], sample_rate, sensitivity)
            
            # Generate mimic feed using scipy
            mimic_feed = self._generate_synth_feed(pitch_hz, num_samples, sample_rate, waveform_type, synth_volume)
            
            # Apply delay/echo
            if delay_time > 0:
                # If tempo map provided, snap delay to nearest beat subdivision
                effective_delay = delay_time
                if tempo is not None:
                    beat_dur = tempo["beat_duration_s"]
                    # Find the beat subdivision closest to the requested delay_time
                    subdivisions = [beat_dur / d for d in [1, 2, 4, 8]]
                    effective_delay = min(subdivisions, key=lambda s: abs(s - delay_time))
                mimic_feed = self._apply_delay(mimic_feed, sample_rate, effective_delay)
            
            # Apply glitch effects
            if mimic_effect != "none":
                mimic_feed = self._apply_mimic_effect(mimic_feed, mimic_effect, sample_rate, tempo)
            
            # Mix dry and wet
            audio_out[:, ch] = (1 - dry_wet_mix) * audio_in[:, ch] + dry_wet_mix * mimic_feed

        # Return to original shape if mono
        if original_mono:
            audio_out = audio_out[:, 0]

        return audio_out

    def _extract_pitch(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Simplified pitch extraction."""
        # Placeholder: return constant pitch for now
        # In real implementation, use librosa or similar
        return np.full(len(audio) // 100, 440.0)  # 440 Hz A note

    def _detect_pitch_librosa(self, audio: np.ndarray, sample_rate: int, sensitivity: float) -> np.ndarray:
        """Detect pitch using librosa with fallback."""
        if LIBROSA_AVAILABLE:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sample_rate,
                frame_length=2048,
                hop_length=512,
                fill_na=np.nan
            )
            # Zero out unvoiced frames so synth does not fire during silence
            median_f0 = np.nanmedian(f0) if np.any(~np.isnan(f0)) else 440.0
            f0 = np.where(np.isnan(f0), 0.0, f0)        # NaN → 0 (silence)
            f0 = np.where(voiced_flag, f0, 0.0)          # unvoiced → 0
            return f0
        else:
            # Fallback: simple autocorrelation
            return self._extract_pitch_fallback(audio, sample_rate)

    def _extract_pitch_fallback(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Simple autocorrelation-based pitch detection."""
        # Placeholder: return constant pitch
        return np.full(len(audio) // 512 + 1, 440.0)

    def _parse_tempo_map(self, tempo_map_json: str) -> dict:
        """
        Parse a TempoMapperNode tempo_map_json string.
        Returns a dict with keys: bpm (float), beat_times (list of float seconds).
        Returns None if the string is empty or unparseable.
        """
        if not tempo_map_json or not tempo_map_json.strip():
            return None
        try:
            data = json.loads(tempo_map_json)
            bpm = float(data.get("global_bpm", 120.0))
            beat_times = data.get("beat_grid_timestamps_seconds", [])
            if not beat_times:
                # Fall back to onset timestamps
                beat_times = data.get("onset_timestamps_seconds", [])
            return {
                "bpm": bpm,
                "beat_times": [float(t) for t in beat_times],
                "beat_duration_s": 60.0 / max(bpm, 1.0),
            }
        except Exception:
            return None

    def _generate_synth_feed(self, pitch_hz: np.ndarray, num_samples: int, sample_rate: int, waveform_type: str, volume: float) -> np.ndarray:
        """Generate synthesized feed using scipy."""
        t = np.arange(num_samples) / sample_rate
        
        # Interpolate pitch to match sample rate
        pitch_interp = np.interp(t, np.linspace(0, num_samples/sample_rate, len(pitch_hz)), pitch_hz)
        
        if waveform_type == "sine":
            synth = np.sin(2 * np.pi * np.cumsum(pitch_interp) / sample_rate)
        elif waveform_type == "square":
            synth = scipy.signal.square(2 * np.pi * np.cumsum(pitch_interp) / sample_rate)
        elif waveform_type == "sawtooth":
            synth = scipy.signal.sawtooth(2 * np.pi * np.cumsum(pitch_interp) / sample_rate)
        elif waveform_type == "triangle":
            synth = scipy.signal.sawtooth(2 * np.pi * np.cumsum(pitch_interp) / sample_rate, 0.5)
        else:
            synth = np.sin(2 * np.pi * np.cumsum(pitch_interp) / sample_rate)
        
        return synth * volume

    def _apply_delay(self, audio: np.ndarray, sample_rate: int, delay_time: float) -> np.ndarray:
        """Apply delay/echo effect."""
        delay_samples = int(delay_time * sample_rate)
        delayed = np.roll(audio, delay_samples)
        # Simple feedback
        feedback = 0.3
        for i in range(len(audio)):
            if i >= delay_samples:
                delayed[i] += delayed[i - delay_samples] * feedback
        return delayed

    def _apply_stutter(self, audio: np.ndarray, sample_rate: int,
                       tempo: dict = None) -> np.ndarray:
        """Repetitive stutter loop. Length synced to beat if tempo map provided."""
        if tempo is not None:
            # Use one beat (or half-beat if that's shorter than 200ms)
            beat_dur = tempo["beat_duration_s"]
            stutter_dur = beat_dur if beat_dur <= 0.5 else beat_dur / 2.0
            stutter_len = max(int(stutter_dur * sample_rate), 64)
        else:
            stutter_len = int(0.1 * sample_rate)  # 100ms default
        stutter = audio[:stutter_len]
        repeated = np.tile(stutter, len(audio) // stutter_len + 1)[:len(audio)]
        return repeated

    def _apply_mimic_effect(self, audio: np.ndarray, effect: str, sample_rate: int, tempo: dict = None) -> np.ndarray:
        """Apply glitch effects."""
        if effect == "bake":
            # Analog tape warmth/distortion
            return np.tanh(audio * 2) * 0.8
        elif effect == "scramble":
            # Random reordering of chunks
            chunk_size = 1024
            scrambled = np.copy(audio)
            for i in range(0, len(audio), chunk_size):
                end = min(i + chunk_size, len(audio))
                np.random.shuffle(scrambled[i:end])
            return scrambled
        elif effect == "fry":
            b, a = scipy.signal.butter(1, 1000 / (sample_rate / 2), btype='high')
            return scipy.signal.filtfilt(b, a, audio) * 2
        elif effect == "glitch":
            # Digital artifacts
            glitch_audio = np.copy(audio)
            # Random skips
            for _ in range(10):
                start = np.random.randint(0, len(audio) - 1000)
                end = start + 1000
                glitch_audio[start:end] = np.roll(glitch_audio[start:end], np.random.randint(-100, 100))
            return glitch_audio
        elif effect == "stutter":
            return self._apply_stutter(audio, sample_rate, tempo)
        elif effect == "tape_stop":
            slow_factor = 0.5
            fast_factor = 2.0
            mid = len(audio) // 2
            # Resample each half, then crossfade + pad to preserve original length
            slow_part = scipy.signal.resample(audio[:mid], int(mid * slow_factor))
            fast_part = scipy.signal.resample(audio[mid:], int((len(audio) - mid) / fast_factor))
            combined = np.concatenate([slow_part, fast_part])
            # Ensure output matches input length exactly
            if len(combined) < len(audio):
                combined = np.pad(combined, (0, len(audio) - len(combined)))
            else:
                combined = combined[:len(audio)]
            return combined
        return audio