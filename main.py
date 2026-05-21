#!/usr/bin/env python3
"""Split an audio file into randomized time-frequency stems."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def decode_to_wav(src: Path, dst: Path) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-acodec",
            "pcm_f32le",
            str(dst),
        ]
    )


def make_band_edges(sr: int, n_fft: int) -> np.ndarray:
    nyquist = sr / 2
    edges = [
        0,
        35,
        60,
        90,
        120,
        150,
        180,
        220,
        270,
        330,
        400,
        480,
        580,
        700,
        840,
        1000,
        1180,
        1380,
        1600,
        1850,
        2150,
        2500,
        2900,
        3350,
        3850,
        4450,
        5200,
        6200,
        7600,
        9500,
        12000,
        15500,
        19000,
        nyquist,
    ]
    edges = np.array([x for x in edges if x <= nyquist], dtype=float)
    if edges[-1] < nyquist:
        edges = np.append(edges, nyquist)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bin_edges = np.unique(np.searchsorted(freqs, edges, side="left"))
    bin_edges[0] = 0
    if bin_edges[-1] != len(freqs):
        bin_edges = np.append(bin_edges, len(freqs))
    return bin_edges.astype(np.int32)


def make_time_edges(
    frames: int,
    sr: int,
    hop_length: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    frame = 0
    while frame < frames:
        seconds = float(rng.uniform(0.12, 0.48))
        span = max(2, int(round(seconds * sr / hop_length)))
        end = min(frames, frame + span)
        edges.append((frame, end))
        frame = end
    return edges


def build_assignment(
    freq_bins: int,
    frames: int,
    band_edges: np.ndarray,
    time_edges: list[tuple[int, int]],
    rng: np.random.Generator,
    tracks: int,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    assignment = np.empty((freq_bins, frames), dtype=np.uint8)
    manifest: list[dict[str, float | int]] = []

    last_by_band = np.full(len(band_edges) - 1, -1, dtype=np.int16)
    for time_index, (t0, t1) in enumerate(time_edges):
        band_order = rng.permutation(len(band_edges) - 1)
        for band_index in band_order:
            f0 = int(band_edges[band_index])
            f1 = int(band_edges[band_index + 1])
            if f0 >= f1:
                continue

            choice = int(rng.integers(0, tracks))
            if choice == last_by_band[band_index]:
                choice = (choice + int(rng.integers(1, tracks))) % tracks
            last_by_band[band_index] = choice
            assignment[f0:f1, t0:t1] = choice

            manifest.append(
                {
                    "time_block": time_index + 1,
                    "freq_band": band_index + 1,
                    "frame_start": t0,
                    "frame_end": t1,
                    "freq_bin_start": f0,
                    "freq_bin_end": f1,
                    "track": choice + 1,
                }
            )

    return assignment, manifest


def write_manifest(
    out_dir: Path,
    manifest: list[dict[str, float | int]],
    freqs: np.ndarray,
    sr: int,
    hop_length: int,
) -> None:
    fieldnames = [
        "time_block",
        "start_sec",
        "end_sec",
        "freq_band",
        "low_hz",
        "high_hz",
        "track",
        "frame_start",
        "frame_end",
        "freq_bin_start",
        "freq_bin_end",
    ]
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest:
            frame_start = int(row["frame_start"])
            frame_end = int(row["frame_end"])
            bin_start = int(row["freq_bin_start"])
            bin_end = int(row["freq_bin_end"])
            writer.writerow(
                {
                    "time_block": row["time_block"],
                    "start_sec": f"{frame_start * hop_length / sr:.6f}",
                    "end_sec": f"{frame_end * hop_length / sr:.6f}",
                    "freq_band": row["freq_band"],
                    "low_hz": f"{float(freqs[bin_start]):.2f}",
                    "high_hz": f"{float(freqs[min(bin_end, len(freqs) - 1)]):.2f}",
                    "track": row["track"],
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "freq_bin_start": bin_start,
                    "freq_bin_end": bin_end,
                }
            )


def encode_mp3(wav_path: Path, mp3_path: Path, bitrate: str) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            str(mp3_path),
        ]
    )


def process(args: argparse.Namespace) -> None:
    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output).expanduser().resolve() if args.output else src.with_name(f"{src.stem}_20tracks_split_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    print(f"[1/7] input={src}", flush=True)
    print(f"[1/7] output={out_dir}", flush=True)
    with tempfile.TemporaryDirectory(prefix="tf20_split_") as tmp:
        tmp_dir = Path(tmp)
        decoded = tmp_dir / "source_f32.wav"
        print("[2/7] decoding to float wav", flush=True)
        decode_to_wav(src, decoded)

        print("[3/7] loading decoded audio", flush=True)
        audio, sr = sf.read(decoded, dtype="float32", always_2d=True)
        audio = audio.T
        channels, samples = audio.shape

        n_fft = args.n_fft
        hop_length = args.hop_length
        print(f"[4/7] computing stft channels={channels} samples={samples} sr={sr} n_fft={n_fft} hop={hop_length}", flush=True)
        stfts = [
            librosa.stft(audio[ch], n_fft=n_fft, hop_length=hop_length, center=True).astype(np.complex64)
            for ch in range(channels)
        ]

        freq_bins, frames = stfts[0].shape
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        band_edges = make_band_edges(sr, n_fft)
        time_edges = make_time_edges(frames, sr, hop_length, rng)
        print(f"[5/7] assigning tf cells freq_bins={freq_bins} frames={frames} bands={len(band_edges) - 1} time_blocks={len(time_edges)} tracks={args.tracks}", flush=True)
        assignment, manifest = build_assignment(freq_bins, frames, band_edges, time_edges, rng, args.tracks)

        print("[6/7] writing metadata and manifest", flush=True)
        write_manifest(out_dir, manifest, freqs, sr, hop_length)
        metadata = {
            "source": str(src),
            "output_dir": str(out_dir),
            "created_at": timestamp,
            "duration_sec": samples / sr,
            "sample_rate": sr,
            "channels": channels,
            "tracks": args.tracks,
            "seed": args.seed,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "time_window_sec_range": [0.12, 0.48],
            "mp3_bitrate": args.bitrate,
            "notes": "Randomized time-frequency split for sound design/private archival use. Vocal-range bands are intentionally finer from roughly 150 Hz to 5.2 kHz.",
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        summary_rows: list[dict[str, str | int | float]] = []
        for track in range(args.tracks):
            print(f"[7/7] rendering track {track + 1:02d}/{args.tracks:02d}", flush=True)
            stems = []
            mask = assignment == track
            for ch in range(channels):
                masked = np.zeros_like(stfts[ch])
                masked[mask] = stfts[ch][mask]
                stem = librosa.istft(masked, hop_length=hop_length, length=samples, center=True)
                stems.append(stem.astype(np.float32))

            stem_audio = np.stack(stems, axis=1)
            peak = float(np.max(np.abs(stem_audio))) if stem_audio.size else 0.0
            scale = 1.0
            if peak > 0.98:
                scale = 0.98 / peak
                stem_audio *= scale

            wav_path = tmp_dir / f"track_{track + 1:02d}.wav"
            mp3_path = out_dir / f"track_{track + 1:02d}.mp3"
            sf.write(wav_path, stem_audio, sr, subtype="FLOAT")
            encode_mp3(wav_path, mp3_path, args.bitrate)

            seconds_active = int(np.any(mask, axis=0).sum()) * hop_length / sr
            summary_rows.append(
                {
                    "track": track + 1,
                    "file": mp3_path.name,
                    "peak_before_scale": f"{peak:.6f}",
                    "scale": f"{scale:.6f}",
                    "assigned_tf_cells": int(mask.sum()),
                    "approx_active_frame_seconds": f"{seconds_active:.3f}",
                }
            )
            print(f"rendered {mp3_path}")

        with (out_dir / "track_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output")
    parser.add_argument("--tracks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--n-fft", type=int, default=4096)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--bitrate", default="320k")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")
    if args.tracks < 2 or args.tracks > 255:
        raise ValueError("--tracks must be between 2 and 255")
    if args.hop_length <= 0 or args.n_fft <= args.hop_length:
        raise ValueError("--n-fft must be greater than --hop-length")

    process(args)


if __name__ == "__main__":
    main()
