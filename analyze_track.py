import argparse, os, sys, glob, json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
import librosa
import pyloudnorm as pyln
from tqdm import tqdm

# ================================
#  Constants / Utilities
# ================================
MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
PITCH_CLASS_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

CAMELOT = {
    "C major":"8B","G major":"9B","D major":"10B","A major":"11B","E major":"12B","B major":"1B",
    "F# major":"2B","Db major":"3B","Ab major":"4B","Eb major":"5B","Bb major":"6B","F major":"7B",
    "A minor":"8A","E minor":"9A","B minor":"10A","F# minor":"11A","C# minor":"12A","G# minor":"1A",
    "D# minor":"2A","A# minor":"3A","F minor":"4A","C minor":"5A","G minor":"6A","D minor":"7A"
}

def percent(x, q):
    x = np.asarray(x)
    return float(np.percentile(x, q)) if x.size else 0.0

def safe_float(x):
    try:
        if hasattr(x, "ndim"):
            x = np.ravel(x)[0]
        return float(x)
    except Exception:
        return 0.0

def hz_mean_spectral_centroid(y, sr):
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    return float(np.mean(cent)) if cent.size else 0.0

def estimate_key_ks_with_score(chroma_12xT: np.ndarray) -> Tuple[str, float]:
    if chroma_12xT.size == 0:
        return "Unknown", 0.0
    chroma_mean = np.mean(chroma_12xT, axis=1)
    s = np.sum(chroma_mean)
    if s > 0:
        chroma_mean = chroma_mean / s

    def rot(profile, k): return np.roll(profile, k)
    best_score, best_key, best_mode = -1e9, "Unknown", "major"
    for k in range(12):
        maj = np.corrcoef(chroma_mean, rot(MAJOR_PROFILE, k))[0,1]
        minr = np.corrcoef(chroma_mean, rot(MINOR_PROFILE, k))[0,1]
        maj = -1 if np.isnan(maj) else maj
        minr = -1 if np.isnan(minr) else minr
        if maj >= minr and maj > best_score:
            best_score, best_key, best_mode = maj, PITCH_CLASS_NAMES[k], "major"
        if minr > maj and minr > best_score:
            best_score, best_key, best_mode = minr, PITCH_CLASS_NAMES[k], "minor"
    conf = float(np.clip((best_score + 1)/2, 0, 1))
    return f"{best_key} {best_mode}", conf

def majority_key_from_chroma(chroma_12xT: np.ndarray, win_frames: int, hop_frames: int):
    T = chroma_12xT.shape[1]
    if T <= 0 or win_frames <= 0:
        return None, 0.0, None
    keys = []
    for s in range(0, max(1, T - win_frames + 1), hop_frames):
        seg = chroma_12xT[:, s:s+win_frames]
        k, _ = estimate_key_ks_with_score(seg)
        keys.append(k)
    if not keys:
        return None, 0.0, None
    cnt = Counter(keys).most_common(2)
    best = cnt[0][0]
    occupancy = cnt[0][1] / max(1, len(keys))
    alt = cnt[1][0] if len(cnt) > 1 else ""
    return best, float(occupancy), alt

def tempo_class(bpm: float) -> str:
    if bpm <= 0: return "unknown"
    if bpm < 85: return "slow"
    if bpm <= 115: return "mid"
    return "fast"

def brightness_class(centroid_hz: float) -> str:
    if centroid_hz <= 0: return "unknown"
    if centroid_hz < 1800: return "warm/dark"
    if centroid_hz <= 2600: return "balanced"
    return "bright/crisp"

def energy_class(score: float) -> str:
    if score < 0.4: return "low"
    if score < 0.7: return "mid"
    return "high"

def build_feel_label(bpm, energy, centroid, rhythm_consistency, valence) -> str:
    t = tempo_class(bpm)
    e = energy_class(energy)
    b = brightness_class(centroid)
    r = "steady" if rhythm_consistency >= 0.75 else "loose"
    mood = "bright" if valence >= 0.6 else ("neutral" if valence >= 0.4 else "moody")
    return f"{mood.capitalize()} · {t} tempo · {e} energy · {r} · {b}"

def calc_confidence(duration_sec, rhythm_consistency, key_conf, beats_count) -> float:
    s_len = np.clip((duration_sec - 60)/60, 0, 1)   # 60~120s에서 0→1
    s_rhy = np.clip(rhythm_consistency, 0, 1)
    s_key = np.clip(key_conf, 0, 1)
    s_beats = np.clip(beats_count/50.0, 0, 1)       # 50개 이상이면 만점
    return float(np.clip(0.35*s_rhy + 0.25*s_key + 0.20*s_len + 0.20*s_beats, 0, 1))

# ================================
# Result schema
# ================================
@dataclass
class TrackFeatures:
    file: str
    duration_sec: float
    tempo_bpm: float
    tempo_class: str
    rhythm_consistency_score: float
    energy_score: float
    valence_score_baseline: float
    song_key: str
    alt_key: str
    camelot: str
    key_confidence: float
    loudness_lufs: float
    loudness_range: float
    mean_centroid: float
    brightness_class: str
    feel_label: str
    confidence_score: float

# ================================
# Core analysis for a single path
# ================================
def analyze_one(path: str) -> TrackFeatures:
    # 1) load
    y, sr = librosa.load(path, sr=44100, mono=True)
    duration_sec = len(y)/sr if sr>0 else 0.0

    # 2) loudness (LRA via 3s segments)
    meter = pyln.Meter(sr)
    try:
        loudness_lufs = meter.integrated_loudness(y)
    except Exception:
        loudness_lufs = 0.0
    seg_len = int(sr * 3)
    segs = [y[i:i+seg_len] for i in range(0, len(y), seg_len) if len(y[i:i+seg_len])>0]
    seg_loud = []
    for seg in segs:
        try:
            seg_loud.append(meter.integrated_loudness(seg))
        except Exception:
            pass
    loudness_range = float(percent(seg_loud, 95) - percent(seg_loud, 10)) if seg_loud else 0.0

    # 3) tempo / beats
    tempo_bpm, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
    tempo_bpm = safe_float(tempo_bpm)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beats_count = len(beat_times)
    if beats_count > 2:
        ibis = np.diff(beat_times)
        cv = np.std(ibis) / (np.mean(ibis) + 1e-9)
        rhythm_consistency_score = float(np.clip(1.0 - cv, 0.0, 1.0))
    else:
        rhythm_consistency_score = 0.0

    # 4) energy (RMS 90/99)
    rmss = librosa.feature.rms(y=y, frame_length=2048, hop_length=512).squeeze()
    r90 = percent(rmss, 90); r99 = percent(rmss, 99) + 1e-9
    energy_score = float(np.clip(r90 / r99, 0.0, 1.0))

    # 5) key (HPSS + tuning + window majority)
    try:
        y_harm, _ = librosa.effects.hpss(y)
        tuning = librosa.estimate_tuning(y=y_harm, sr=sr)
        hop_length = 512
        chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, n_chroma=12,
                                            hop_length=hop_length, tuning=tuning)
        song_key, key_conf = estimate_key_ks_with_score(chroma)
        win_frames = int((8.0 * sr) / hop_length)
        hop_frames = int((4.0 * sr) / hop_length)
        best_key, occ_conf, alt_key = majority_key_from_chroma(chroma, win_frames, hop_frames)
        if best_key:
            key_conf = float(np.clip((key_conf + occ_conf)/2.0, 0.0, 1.0))
            song_key = best_key
        else:
            alt_key = ""
    except Exception:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12)
        song_key, key_conf = estimate_key_ks_with_score(chroma)
        alt_key = ""

    camelot = CAMELOT.get(song_key, "")

    # 6) spectral centroid & brightness class
    mean_centroid = hz_mean_spectral_centroid(y, sr)
    bclass = brightness_class(mean_centroid)

    # 7) baseline valence (heuristic)
    val_brightness = np.clip(mean_centroid/8000.0, 0.0, 1.0)
    val_tempo = np.clip((tempo_bpm-60.0)/(180.0-60.0), 0.0, 1.0) if tempo_bpm>0 else 0.0
    valence_score_baseline = float(np.clip(
        0.45*val_brightness + 0.35*rhythm_consistency_score + 0.20*val_tempo, 0.0, 1.0
    ))

    # 8) human-readable label & confidence
    tclass = tempo_class(tempo_bpm)
    feel_label = build_feel_label(tempo_bpm, energy_score, mean_centroid,
                                  rhythm_consistency_score, valence_score_baseline)
    confidence = calc_confidence(duration_sec, rhythm_consistency_score, key_conf, beats_count)

    return TrackFeatures(
        file=os.path.basename(path),
        duration_sec=float(duration_sec),
        tempo_bpm=float(tempo_bpm),
        tempo_class=tclass,
        rhythm_consistency_score=float(rhythm_consistency_score),
        energy_score=float(energy_score),
        valence_score_baseline=float(valence_score_baseline),
        song_key=song_key,
        alt_key=alt_key,
        camelot=camelot,
        key_confidence=float(key_conf),
        loudness_lufs=float(loudness_lufs),
        loudness_range=float(loudness_range),
        mean_centroid=float(mean_centroid),
        brightness_class=bclass,
        feel_label=feel_label,
        confidence_score=float(confidence)
    )

# ================================
# Worker wrapper (for Pool)
# ================================
def worker_analyze(path: str) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """Return (path, row_dict or None on error, error_msg or None)."""
    try:
        feats = analyze_one(path)
        return path, asdict(feats), None
    except Exception as e:
        return path, None, str(e)

# ================================
# Main
# ================================
def main():
    parser = argparse.ArgumentParser(description="Sona: Audio Analyzer (KeyStable + Multiprocessing + Progress)")
    parser.add_argument("input", help="분석할 파일(.wav/.mp3 등) 또는 폴더")
    parser.add_argument("--outdir", default="analysis_out", help="결과 저장 폴더")
    parser.add_argument("--pattern", default="*.mp3,*.wav,*.m4a,*.flac", help="폴더 입력 시 확장자 패턴(쉼표 구분)")
    parser.add_argument("--jobs", type=int, default=max(1, (cpu_count() or 2) - 1), help="동시 처리 프로세스 수")
    parser.add_argument("--skip-existing", action="store_true", help="이미 분석된 JSON이 있으면 건너뛰기")
    
    # === [수정된 부분] 이 줄이 추가되었습니다 ===
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Collect paths
    if os.path.isdir(args.input):
        exts = [p.strip() for p in args.pattern.split(",") if p.strip()]
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(args.input, "**", ext), recursive=True))
    else:
        paths = [args.input]

    # Skip already-done
    if args.skip_existing:
        filtered = []
        for p in paths:
            base = os.path.splitext(os.path.basename(p))[0]
            json_path = os.path.join(args.outdir, f"{base}.json")
            if not os.path.exists(json_path):
                filtered.append(p)
        paths = filtered

    if not paths:
        print("No input files to process.")
        return

    rows: List[Dict[str, Any]] = []

    # Multiprocessing pool with progress bar
    jobs = max(1, args.jobs)
    print(f"Using {jobs} process(es). Files to analyze: {len(paths)}")

    with Pool(processes=jobs) as pool:
        results = list(tqdm(pool.imap_unordered(worker_analyze, paths),
                                   total=len(paths), desc="Analyzing", unit="file"))

    for path, row, err in results:
        base = os.path.splitext(os.path.basename(path))[0]
        if err is None and row is not None:
            json_path = os.path.join(args.outdir, f"{base}.json")
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(row, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARN] JSON write failed for {path}: {e}", file=sys.stderr)
            rows.append(row)
        else:
            print(f"\n[ERR] {path} → {err}", file=sys.stderr)

    if not rows:
        print("No results. Check your input path/codecs.")
        return

    # Save CSV/XLSX with timestamp (KST)
    df = pd.DataFrame(rows)
    ts = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    csv_path  = os.path.join(args.outdir, f"summary_{ts}.csv")
    xlsx_path = os.path.join(args.outdir, f"summary_{ts}.xlsx")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="tracks")

    print(f"\nSaved CSV : {csv_path}")
    print(f"Saved XLSX: {xlsx_path}")

if __name__ == "__main__":
    main()