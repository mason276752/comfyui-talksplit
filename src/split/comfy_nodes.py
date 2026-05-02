"""ComfyUI custom node wrappers.

Granular nodes (Sentences -> Embed -> Score -> Optimize -> Assemble) for
flexible workflows, plus an Auto node that runs the whole pipeline.
"""
from __future__ import annotations

import io
import math
import re

from .boundary import block_similarity, depth_scores, threshold_for_sensitivity
from .embedder import Embedder
from .markers import DEFAULT_MARKERS, boost_depths, parse_markers
from .optimizer import optimize_boundaries
from .segmenter import SOFT_COMMA, Sentence, normalize_paragraph, split_sentences

_T_SENTENCES = "TS_SENTENCES"
_T_EMBEDDINGS = "TS_EMBEDDINGS"
_T_DEPTHS = "TS_DEPTHS"
_T_SPLITS = "TS_SPLITS"

_CATEGORY = "talksplit"


class TalksplitSentences:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "fallback_chunk": ("INT", {"default": 30, "min": 5, "max": 500, "step": 1}),
                "clause_level": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (_T_SENTENCES, "STRING")
    RETURN_NAMES = ("sentences", "text")
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, text: str, fallback_chunk: int, clause_level: bool):
        sents = split_sentences(
            text,
            fallback_chunk=fallback_chunk,
            extra_terminators=SOFT_COMMA if clause_level else "",
        )
        return (sents, text)


class TalksplitEmbed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sentences": (_T_SENTENCES,),
                "model_name": ("STRING", {"default": "BAAI/bge-m3"}),
            },
            "optional": {
                "device": ("STRING", {"default": ""}),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 256}),
            },
        }

    RETURN_TYPES = (_T_EMBEDDINGS,)
    RETURN_NAMES = ("embeddings",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, sentences, model_name: str, device: str = "", batch_size: int = 32):
        if not sentences:
            return (None,)
        embedder = _get_embedder(model_name, device)
        embs = embedder.embed([s.text for s in sentences], batch_size=batch_size)
        return (embs,)


class TalksplitScore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"embeddings": (_T_EMBEDDINGS,)},
            "optional": {
                "block_size": ("INT", {"default": 1, "min": 1, "max": 20,
                    "tooltip": "Sentences per side for block comparison. "
                               "Increase (e.g. 3) for articles with gradual topic transitions."}),
            },
        }

    RETURN_TYPES = (_T_DEPTHS,)
    RETURN_NAMES = ("depths",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, embeddings, block_size: int = 1):
        sim = block_similarity(embeddings, block_size=block_size)
        depths = depth_scores(sim)
        # bundle is (sim, depths, raw_depths_for_threshold). raw == depths until
        # something post-processes (e.g. MarkerBoost) and wants the threshold to
        # stay anchored to the natural distribution.
        return ((sim, depths, depths),)


class TalksplitMarkerBoost:
    """Boost depth scores at gaps preceding sentences that begin with a
    discourse marker like 'Let me switch' or '接下來'."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sentences": (_T_SENTENCES,),
                "depths": (_T_DEPTHS,),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "custom_markers": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Newline / comma-separated phrases. Leave empty to use defaults.",
                }),
            },
        }

    RETURN_TYPES = (_T_DEPTHS,)
    RETURN_NAMES = ("depths",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, sentences, depths, sensitivity: float = 1.0, custom_markers: str = ""):
        sim, dps, raw = _unpack_depths(depths)
        markers = parse_markers(custom_markers) if custom_markers.strip() else DEFAULT_MARKERS
        threshold = threshold_for_sensitivity(raw, sensitivity)
        boosted = boost_depths(sentences, dps, markers=markers, threshold=threshold)
        return ((sim, boosted, raw),)


class TalksplitOptimize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depths": (_T_DEPTHS,),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "min_sentences": ("INT", {"default": 2, "min": 1, "max": 200}),
                "max_sentences": ("INT", {"default": 15, "min": 1, "max": 500}),
                "target_paragraphs": ("INT", {"default": 0, "min": 0, "max": 200}),
            }
        }

    RETURN_TYPES = (_T_SPLITS,)
    RETURN_NAMES = ("splits",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, depths, sensitivity, min_sentences, max_sentences, target_paragraphs):
        _sim, dps, raw = _unpack_depths(depths)
        target = target_paragraphs if target_paragraphs > 0 else None
        threshold = threshold_for_sensitivity(raw, sensitivity) if raw is not dps else None
        splits = optimize_boundaries(
            dps,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            sensitivity=sensitivity,
            target_paragraphs=target,
            threshold=threshold,
        )
        return (splits,)


class TalksplitAssemble:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True}),
                "sentences": (_T_SENTENCES,),
                "splits": (_T_SPLITS,),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("paragraphs",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True

    def run(self, text, sentences, splits):
        out = _assemble(text, sentences, splits)
        return {"ui": {"text": [out]}, "result": (out,)}


class TalksplitPlot:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depths": (_T_DEPTHS,),
                "splits": (_T_SPLITS,),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("plot",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, depths, splits, sensitivity):
        sim, dps, raw = _unpack_depths(depths)
        threshold = threshold_for_sensitivity(raw, sensitivity)
        return (_plot_to_tensor(sim, dps, splits, threshold),)


class TalksplitSplitToList:
    """Split the joined paragraphs STRING into individual items.

    Marked ``OUTPUT_IS_LIST`` so any downstream node (e.g. CLIPTextEncode)
    runs once per paragraph — the standard ComfyUI pattern for per-item
    iteration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "paragraphs": ("STRING", {"multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("paragraph", "index")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, paragraphs: str):
        items = [p.strip() for p in paragraphs.split("\n\n") if p.strip()]
        return (items, list(range(len(items))))


class TalksplitPickParagraph:
    """Pick one paragraph from the joined STRING by index.

    Negative indices count from the end (-1 = last). Out-of-range indices
    clamp to the valid range. ``count`` reports the total number of
    paragraphs available so downstream nodes can size loops correctly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "paragraphs": ("STRING", {"multiline": True, "forceInput": True}),
                "index": ("INT", {"default": 0, "min": -999, "max": 999}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("paragraph", "count")
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, paragraphs: str, index: int):
        items = [p.strip() for p in paragraphs.split("\n\n") if p.strip()]
        if not items:
            return ("", 0)
        n = len(items)
        idx = index if index >= 0 else n + index
        idx = max(0, min(idx, n - 1))
        return (items[idx], n)


class TalksplitCleanForTTS:
    """Remove characters that confuse TTS engines: brackets, special quotes,
    colons, ellipses, dashes, and other non-speech symbols.
    Natural speech punctuation (。！？，；、) is preserved."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, text: str):
        return (_clean_for_tts(text),)


# Step 1: characters that map to a pause rather than being deleted outright,
# so clause rhythm is preserved in the output audio.
_TTS_PAUSE_RE = re.compile(
    r"[\u2026\u22ef\u2025]+"          # … ⋯ ‥  (ellipsis variants)
    r"|[\u2014\u2015\u2013\u2012]+"   # — ― – ‒  (dash variants)
    r"|[\uff1a\u003a\uff0f\u002f]"    # ：: /／  (colon / slash)
)

# Step 2a: convert enclosed/circled numbers (①→1, ⑩→10, ⑴→1 …) to digits
# so "第①類" → "第1類" rather than "第類".
_ENCLOSED_NUM: dict[int, str] = {}
for _i, _base in enumerate(range(0x2460, 0x2474), 1):   # ①–⑳  (1–20)
    _ENCLOSED_NUM[_base] = str(_i)
for _i, _base in enumerate(range(0x2474, 0x2488), 1):   # ⑴–⒇  (1–20)
    _ENCLOSED_NUM[_base] = str(_i)
for _i, _base in enumerate(range(0x2488, 0x249c), 1):   # ⒈–⒛  (1–20)
    _ENCLOSED_NUM[_base] = str(_i)
for _i, _base in enumerate(range(0x24b6, 0x24d0), 65):  # Ⓐ–Ⓩ  → A–Z
    _ENCLOSED_NUM[_base] = chr(_i)
for _i, _base in enumerate(range(0x24d0, 0x24ea), 97):  # ⓐ–ⓩ  → a–z
    _ENCLOSED_NUM[_base] = chr(_i)


def _normalize_enclosed(text: str) -> str:
    out = []
    for ch in text:
        out.append(_ENCLOSED_NUM.get(ord(ch), ch))
    return "".join(out)


# Step 2b: full-width ASCII (Ａ→A, １→1) so the whitelist below catches them.
def _halfwidth(text: str) -> str:
    out = []
    for ch in text:
        cp = ord(ch)
        if 0xFF01 <= cp <= 0xFF5E:   # full-width ASCII variants block
            out.append(chr(cp - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)

# Step 3: whitelist — anything NOT matching is stripped.
# Kept: CJK characters, Latin letters, digits, spaces, newlines,
#       and the small set of punctuation that TTS reads as natural pauses.
_TTS_SAFE_RE = re.compile(
    r"[^"
    r"\u4e00-\u9fff"        # CJK Unified Ideographs (main block)
    r"\u3400-\u4dbf"        # CJK Extension A
    r"\uf900-\ufaff"        # CJK Compatibility Ideographs
    r"A-Za-z0-9 \n"         # Latin, digits, space, newline
    r"\u3002"               # 。
    r"\uff01"               # ！
    r"\uff1f"               # ？
    r"\uff0c"               # ，
    r"\uff1b"               # ；
    r"\u3001"               # 、
    r".!?,;"                # ASCII equivalents
    r"]+"
)

_MULTI_PAUSE_RE = re.compile(r"[，,、]{2,}")


def _clean_for_tts(text: str) -> str:
    text = _TTS_PAUSE_RE.sub("，", text)    # preserve rhythm at clause breaks
    text = _normalize_enclosed(text)        # ①→1, Ⓐ→A before whitelist
    text = _halfwidth(text)                 # Ａ→A, １→1
    text = _TTS_SAFE_RE.sub("", text)       # strip everything not on whitelist
    text = _MULTI_PAUSE_RE.sub("，", text)  # collapse consecutive pauses
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


class TalksplitTrimSilence:
    """移除 AUDIO 中的靜音段（頭、尾、及超過 min_silence_ms 的中間靜音）。

    用途：TTS 有時在段落中間未生成內容，留下大段假靜音；
    本節點移除這些假靜音，讓圖片時長精確對齊實際語音。
    自然停頓（< min_silence_ms）保持不動。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "threshold_db": ("FLOAT", {"default": -40.0, "min": -80.0, "max": -10.0, "step": 1.0}),
                "pad_ms": ("INT", {"default": 50, "min": 0, "max": 500, "step": 10,
                    "tooltip": "在每段語音邊緣保留的緩衝（毫秒），避免截掉起音/收音。"}),
                "min_silence_ms": ("INT", {"default": 500, "min": 50, "max": 5000, "step": 50,
                    "tooltip": "短於此長度的靜音視為自然停頓，不移除。預設 500ms。"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, audio, threshold_db: float, pad_ms: int, min_silence_ms: int):
        import torch
        waveform = audio["waveform"]          # [1, C, T]
        sr = audio["sample_rate"]
        mono = waveform[0].abs().max(dim=0).values  # [T]
        threshold = 10 ** (threshold_db / 20)
        mask = (mono > threshold).float()
        if mask.sum() == 0:
            return (audio,)

        T = waveform.shape[2]
        pad = int(pad_ms / 1000 * sr)
        min_sil = int(min_silence_ms / 1000 * sr)

        # 用 diff 向量化偵測語音起止邊界（不用 Python 逐樣本迴圈）
        # mask_ext: 在頭尾各補一個 0，方便偵測首尾邊界
        mask_ext = torch.cat([mask.new_zeros(1), mask, mask.new_zeros(1)])
        diff = mask_ext[1:] - mask_ext[:-1]   # 長度 T+1
        # diff[i]=+1 → sample i 開始有聲；diff[i]=-1 → sample i 開始無聲
        starts = (diff > 0).nonzero(as_tuple=True)[0].tolist()
        ends   = (diff < 0).nonzero(as_tuple=True)[0].tolist()
        # 此時 segment k 的有聲範圍為 [starts[k], ends[k])

        # 合併間距小於 min_sil 的相鄰片段（保留自然停頓）
        merged_s = [starts[0]]
        merged_e = [ends[0]]
        for s, e in zip(starts[1:], ends[1:]):
            if s - merged_e[-1] < min_sil:
                merged_e[-1] = e          # 合併：延伸前一段的結尾
            else:
                merged_s.append(s)
                merged_e.append(e)

        # 套用 pad，拼接所有語音片段
        chunks = []
        for s, e in zip(merged_s, merged_e):
            cs = max(0, s - pad)
            ce = min(T, e + pad)
            chunks.append(waveform[:, :, cs:ce])
        combined = torch.cat(chunks, dim=2)
        return ({"waveform": combined, "sample_rate": sr},)


class TalksplitConcatAudio:
    """Concatenate a list of AUDIO objects along the time axis (dim=2).

    VHS_Unbatch uses torch.cat(dim=0) which fails when audio segments have
    different lengths. This node concatenates along the samples dimension.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    INPUT_IS_LIST = True
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, audio):
        import torch
        waveforms = [a["waveform"] for a in audio]
        combined = torch.cat(waveforms, dim=2)
        return ({"waveform": combined, "sample_rate": audio[0]["sample_rate"]},)


class TalksplitRepeatImageForAudio:
    """Repeat a single image for exactly as many frames as the audio duration requires.

    Designed to be used inside a per-paragraph list loop:
      TalksplitSplitToList → ... → IMAGE_i  ┐
                                   AUDIO_i  ┴→ TalksplitRepeatImageForAudio → [n_frames, H, W, C]

    Then VHS_Unbatch collects all per-paragraph frame batches and concatenates them,
    and TalksplitConcatAudio does the same for audio, before feeding into VHS_VideoCombine.

    zoom_start / zoom_end: Ken Burns effect. 1.0 = original, 1.2 = 20% zoomed in.
    Setting both to 1.0 disables zoom (default, no effect).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "zoom_start": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.01}),
                "zoom_end":   ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "index": ("INT", {"default": 0, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("SEGMENT_VIDEO",)
    RETURN_NAMES = ("segment_path",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, image, audio, fps: int, zoom_start: float, zoom_end: float, index: int = 0):
        import os
        import subprocess
        import tempfile
        import uuid
        import torch
        import torch.nn.functional as F

        duration = audio["waveform"].size(2) / audio["sample_rate"]
        n_frames = max(1, math.ceil(duration * fps))

        # image shape: [B, H, W, C] — take first frame, clamp to [0,1]
        frame = image[0].cpu().clamp(0.0, 1.0)  # [H, W, C]
        H, W, C = frame.shape

        # h264 requires even dimensions — crop 1px if needed
        H_enc = H if H % 2 == 0 else H - 1
        W_enc = W if W % 2 == 0 else W - 1
        if H_enc != H or W_enc != W:
            frame = frame[:H_enc, :W_enc, :]
            H, W = H_enc, W_enc

        # 奇數段落交換 zoom 方向（交替 zoom in / zoom out）
        if zoom_start != zoom_end and index % 2 == 1:
            zoom_start, zoom_end = zoom_end, zoom_start

        path = os.path.join(tempfile.gettempdir(), f"talksplit_seg_{uuid.uuid4().hex}.mp4")

        # Pipe raw RGB24 frames to ffmpeg → h264 mp4 segment.
        # Peak RAM = 1 frame (~12 MB for 1024²) regardless of n_frames.
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{W}x{H}",
            "-framerate", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "fast",
            "-movflags", "+faststart",
            path,
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg and make sure it is accessible.")

        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(n_frames)
        except Exception:
            pbar = None

        try:
            denom = max(n_frames - 1, 1)
            if zoom_start == zoom_end:
                # No zoom: compute frame bytes once, repeat
                frame_bytes = (frame * 255).to(torch.uint8).numpy().tobytes()
                for _ in range(n_frames):
                    proc.stdin.write(frame_bytes)
                    if pbar:
                        pbar.update(1)
            else:
                # Ken Burns: compute one zoomed frame at a time, write immediately
                for i in range(n_frames):
                    zoom = zoom_start + (zoom_end - zoom_start) * (i / denom)
                    crop_h = max(1, round(H / zoom))
                    crop_w = max(1, round(W / zoom))
                    top  = (H - crop_h) // 2
                    left = (W - crop_w) // 2
                    t_in = frame[top:top + crop_h, left:left + crop_w, :].permute(2, 0, 1).unsqueeze(0)
                    f = F.interpolate(t_in, size=(H, W), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0)
                    proc.stdin.write((f * 255).to(torch.uint8).numpy().tobytes())
                    if pbar:
                        pbar.update(1)

            _, stderr_data = proc.communicate()  # closes stdin, waits, reads stderr
        except Exception:
            proc.kill()
            proc.wait()
            raise

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg segment encoding failed:\n{stderr_data.decode(errors='replace')}")

        return (path,)


class TalksplitConcatImages:
    """Collect per-paragraph temp frame files and concatenate them into one IMAGE tensor.

    Designed to replace VHS_Unbatch in the video pipeline.  Loads each segment
    sequentially and deletes the temp file immediately after loading, so peak
    memory equals (accumulated result) + (one segment), rather than all segments
    simultaneously.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"frames_path": ("TEMP_FRAMES",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    INPUT_IS_LIST = True
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, frames_path):
        import os
        import torch

        result = None
        for path in frames_path:
            chunk = torch.load(path, map_location="cpu", weights_only=True)
            result = chunk if result is None else torch.cat([result, chunk], dim=0)
            os.remove(path)

        return (result,)


class TalksplitBuildVideo:
    """Collect per-paragraph video segments and audio, then produce a single MP4.

    Replaces TalksplitConcatImages + TalksplitConcatAudio + VHS_VideoCombine with
    a streaming pipeline:

      1. ffmpeg concat demuxer  — joins segment .mp4 files via bitstream copy
         (no decode; negligible RAM and CPU)
      2. torchaudio.save        — writes concatenated audio to a temp WAV
         (peak RAM = total audio, typically < 50 MB)
      3. ffmpeg mux             — combines video + audio into the final .mp4

    Peak RAM during the entire node: ~total audio size (tiny).
    Disk peak: all segment files + 1 intermediate video + 1 wav + 1 final output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment": ("SEGMENT_VIDEO",),
                "audio":   ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "talksplit_video"}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "save_output": ("BOOLEAN", {"default": False}),
                "transition_duration": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 3.0, "step": 0.1,
                    "tooltip": "Crossfade duration between segments (seconds). 0 = hard cut (fast bitstream copy)."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, segment, audio, filename_prefix, frame_rate=24, save_output=False, transition_duration=0.5):
        import os
        import subprocess
        import tempfile
        import uuid
        import torch
        import torchaudio

        # INPUT_IS_LIST wraps widget values in a list too
        prefix               = filename_prefix[0]       if isinstance(filename_prefix, list)       else filename_prefix
        frame_rate           = frame_rate[0]            if isinstance(frame_rate, list)            else frame_rate
        save_output          = save_output[0]           if isinstance(save_output, list)           else save_output
        transition_duration  = transition_duration[0]   if isinstance(transition_duration, list)   else transition_duration

        try:
            import folder_paths
            if save_output:
                dest_dir = folder_paths.get_output_directory()
                dest_type = "output"
            else:
                dest_dir = folder_paths.get_temp_directory()
                dest_type = "temp"
        except ImportError:
            dest_dir = tempfile.gettempdir()
            dest_type = "temp"

        tmp   = tempfile.gettempdir()
        rid   = uuid.uuid4().hex
        concat_list  = os.path.join(tmp, f"talksplit_concat_{rid}.txt")
        merged_video = os.path.join(tmp, f"talksplit_merged_{rid}.mp4")
        audio_wav    = os.path.join(tmp, f"talksplit_audio_{rid}.wav")
        final_video  = os.path.join(dest_dir, f"{prefix}_{rid[:8]}.mp4")

        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(3)
        except Exception:
            pbar = None

        try:
            # ── Step 1: concat / xfade video segments ────────────────────────
            if transition_duration > 0.0 and len(segment) > 1:
                # xfade crossfade: query each segment duration, then chain with xfade filter
                durations = [_get_video_duration(seg) for seg in segment]
                fc, out_label = _build_xfade_filterchain(durations, transition_duration)
                input_args = []
                for seg_path in segment:
                    input_args += ["-i", seg_path]
                result = subprocess.run(
                    ["ffmpeg", "-y"] + input_args + [
                        "-filter_complex", fc,
                        "-map", out_label,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "18", "-preset", "fast",
                        merged_video,
                    ],
                    capture_output=True,
                )
            else:
                # Hard cut: bitstream copy concat (no re-encode, fast)
                with open(concat_list, "w") as f:
                    for seg_path in segment:
                        f.write(f"file '{seg_path}'\n")
                result = subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                     "-i", concat_list, "-c", "copy", merged_video],
                    capture_output=True,
                )

            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg video combine failed:\n{result.stderr.decode(errors='replace')}")

            # Delete segment files now that they are merged
            for seg_path in segment:
                try:
                    os.remove(seg_path)
                except OSError:
                    pass
            if pbar:
                pbar.update(1)

            # ── Step 2: concatenate audio tensors along time axis (dim=2) ──────
            waveforms = [a["waveform"] for a in audio]
            combined  = torch.cat(waveforms, dim=2)  # [1, C, total_T]
            sr        = audio[0]["sample_rate"]

            # xfade 每個過渡從影片扣掉 F 秒，音訊保持原長 → 結尾對不齊。
            # 提前將音訊截到與影片等長，避免最後一段語音被 -shortest 截斷。
            if transition_duration > 0.0 and len(segment) > 1:
                video_dur = sum(durations) - (len(segment) - 1) * transition_duration
                max_samples = int(video_dur * sr)
                if combined.shape[2] > max_samples:
                    combined = combined[:, :, :max_samples]

            torchaudio.save(audio_wav, combined[0], sr)  # [C, T]
            del waveforms, combined
            if pbar:
                pbar.update(1)

            # ── Step 3: mux audio into video ──────────────────────────────────
            result = subprocess.run(
                ["ffmpeg", "-y",
                 "-i", merged_video,
                 "-i", audio_wav,
                 "-c:v", "copy",
                 "-c:a", "aac", "-b:a", "192k",
                 "-shortest",
                 final_video],
                capture_output=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg mux failed:\n{result.stderr.decode(errors='replace')}")
            if pbar:
                pbar.update(1)

        finally:
            for p in (concat_list, merged_video, audio_wav):
                try:
                    os.remove(p)
                except OSError:
                    pass

        print(f"[TalksplitBuildVideo] saved → {final_video}")
        return {
            "ui": {"gifs": [{"filename": os.path.basename(final_video),
                             "subfolder": "",
                             "type": dest_type,
                             "format": "video/h264-mp4",
                             "frame_rate": frame_rate}]},
            "result": (final_video,),
        }


class TalksplitAuto:
    """One-shot node: text in, paragraphs out."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "min_sentences": ("INT", {"default": 2, "min": 1, "max": 200}),
                "max_sentences": ("INT", {"default": 15, "min": 1, "max": 500}),
                "target_paragraphs": ("INT", {"default": 0, "min": 0, "max": 200}),
                "model_name": ("STRING", {"default": "BAAI/bge-m3"}),
                "use_markers": ("BOOLEAN", {"default": True}),
                "clause_level": ("BOOLEAN", {"default": False}),
                "block_size": ("INT", {"default": 1, "min": 1, "max": 20,
                    "tooltip": "Sentences per side for block comparison. "
                               "Increase (e.g. 3) for articles with gradual topic transitions."}),
            },
            "optional": {
                "device": ("STRING", {"default": ""}),
                "fallback_chunk": ("INT", {"default": 30, "min": 5, "max": 500}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("paragraphs",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True

    def run(self, text, sensitivity, min_sentences, max_sentences, target_paragraphs,
            model_name, use_markers, clause_level, block_size,
            device: str = "", fallback_chunk: int = 30):
        sentences = split_sentences(
            text,
            fallback_chunk=fallback_chunk,
            extra_terminators=SOFT_COMMA if clause_level else "",
        )
        if len(sentences) < 2:
            return {"ui": {"text": [text]}, "result": (text,)}
        embedder = _get_embedder(model_name, device)
        embs = embedder.embed([s.text for s in sentences])
        sim = block_similarity(embs, block_size=block_size)
        dps = depth_scores(sim)
        raw_threshold = threshold_for_sensitivity(dps, sensitivity)
        if use_markers:
            dps = boost_depths(sentences, dps, threshold=raw_threshold)
        target = target_paragraphs if target_paragraphs > 0 else None
        splits = optimize_boundaries(
            dps,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            sensitivity=sensitivity,
            target_paragraphs=target,
            threshold=raw_threshold if use_markers else None,
        )
        out = _assemble(text, sentences, splits)
        return {"ui": {"text": [out]}, "result": (out,)}


def _unpack_depths(bundle):
    """Unpack the DEPTHS socket bundle. Older 2-tuples (sim, depths) are
    accepted for backward compatibility."""
    if len(bundle) == 2:
        sim, dps = bundle
        return sim, dps, dps
    sim, dps, raw = bundle
    return sim, dps, raw


# LRU-1 embedder slot: keeps only the most recently used model loaded.
# Switching to a different (model_name, device) pair evicts the previous
# one so its SentenceTransformer weights (~2 GB for bge-m3) are released.
_embedder_cache: dict[tuple[str, str], Embedder] = {}


def _get_embedder(model_name: str, device: str) -> Embedder:
    key = (model_name, device or "")
    if key in _embedder_cache:
        return _embedder_cache[key]
    # Evict every existing entry before loading the new model.
    _embedder_cache.clear()
    _embedder_cache[key] = Embedder(model_name=model_name, device=device or None)
    return _embedder_cache[key]


def _assemble(text: str, sentences: list[Sentence], splits: list[int]) -> str:
    if not sentences:
        return ""
    bounds = [-1] + sorted(splits) + [len(sentences) - 1]
    paras: list[str] = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i] + 1, bounds[i + 1]
        if start > end:
            continue
        seg = sentences[start:end + 1]
        paras.append(normalize_paragraph(text[seg[0].start : seg[-1].end]))
    return "\n\n".join(paras)


def _plot_to_tensor(sim, depths, splits, threshold):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    x = np.arange(len(sim))
    ax1.plot(x, sim, color="steelblue", label="similarity")
    ax1.set_ylabel("similarity")
    ax1.legend(loc="upper right")
    ax2.plot(x, depths, color="darkorange", label="depth")
    ax2.axhline(threshold, color="gray", linestyle="--", label=f"threshold={threshold:.3f}")
    ax2.set_ylabel("depth")
    ax2.set_xlabel("gap index")
    ax2.legend(loc="upper right")
    for g in splits:
        ax1.axvline(g, color="crimson", alpha=0.4)
        ax2.axvline(g, color="crimson", alpha=0.4)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    buf.close()  # convert() copied pixels; buf no longer needed
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _get_video_duration(path: str) -> float:
    """Return the duration of a video file in seconds using ffprobe."""
    import subprocess
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def _build_xfade_filterchain(durations: list, fade_dur: float) -> tuple:
    """Build an ffmpeg filter_complex string that chains xfade across N video inputs.

    Each input is labelled [0:v], [1:v], ... as usual.
    Returns (filter_complex_string, output_label).

    Offset formula: for transition between segment i and i+1,
      offset_i = sum(durations[0..i]) - (i+1) * fade_dur
    This ensures the crossfade starts fade_dur seconds before segment i ends.
    """
    N = len(durations)
    if N == 1:
        return "[0:v]null[vout]", "[vout]"

    parts = []
    cumulative = 0.0
    prev = "[0:v]"
    for i in range(N - 1):
        cumulative += durations[i]
        offset = max(0.0, cumulative - fade_dur * (i + 1))
        out = "[vout]" if i == N - 2 else f"[v{i + 1}]"
        parts.append(
            f"{prev}[{i + 1}:v]xfade=transition=fade:duration={fade_dur:.3f}:offset={offset:.4f}{out}"
        )
        prev = out
    return ";".join(parts), "[vout]"


NODE_CLASS_MAPPINGS = {
    "TalksplitSentences": TalksplitSentences,
    "TalksplitEmbed": TalksplitEmbed,
    "TalksplitScore": TalksplitScore,
    "TalksplitMarkerBoost": TalksplitMarkerBoost,
    "TalksplitOptimize": TalksplitOptimize,
    "TalksplitAssemble": TalksplitAssemble,
    "TalksplitPlot": TalksplitPlot,
    "TalksplitAuto": TalksplitAuto,
    "TalksplitSplitToList": TalksplitSplitToList,
    "TalksplitPickParagraph": TalksplitPickParagraph,
    "TalksplitCleanForTTS": TalksplitCleanForTTS,
    "TalksplitTrimSilence": TalksplitTrimSilence,
    "TalksplitConcatAudio": TalksplitConcatAudio,
    "TalksplitRepeatImageForAudio": TalksplitRepeatImageForAudio,
    "TalksplitConcatImages": TalksplitConcatImages,
    "TalksplitBuildVideo": TalksplitBuildVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TalksplitSentences": "Talksplit · Sentences",
    "TalksplitEmbed": "Talksplit · Embed",
    "TalksplitScore": "Talksplit · Score",
    "TalksplitMarkerBoost": "Talksplit · Marker Boost",
    "TalksplitOptimize": "Talksplit · Optimize",
    "TalksplitAssemble": "Talksplit · Assemble",
    "TalksplitPlot": "Talksplit · Plot",
    "TalksplitAuto": "Talksplit · Auto",
    "TalksplitSplitToList": "Talksplit · Split to List",
    "TalksplitPickParagraph": "Talksplit · Pick Paragraph",
    "TalksplitCleanForTTS": "Talksplit · Clean for TTS",
    "TalksplitTrimSilence": "Talksplit · Trim Silence",
    "TalksplitConcatAudio": "Talksplit · Concat Audio",
    "TalksplitRepeatImageForAudio": "Talksplit · Repeat Image for Audio",
    "TalksplitConcatImages": "Talksplit · Concat Images",
    "TalksplitBuildVideo": "Talksplit · Build Video",
}
