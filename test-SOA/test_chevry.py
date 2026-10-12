"""Extrait Chevry puis appel direct des fonctions de Volleyball-Metrics.

Depuis la racine : python test-SOA/test_chevry.py --mode extract
Voir README.md pour la venv et les checkpoints indispensables.
"""

import argparse
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BACKEND = ROOT / "Volleyball-Metrics" / "Backend"
ANALYSIS = BACKEND / "Analysis"


def preflight():
    """Diagnostic sans charger les poids ni telecharger de modele."""
    missing = []
    for name in ("cv2", "imageio_ffmpeg", "torch", "torchvision", "ultralytics",
                 "transformers", "safetensors", "onnxruntime", "lap"):
        ok = importlib.util.find_spec(name) is not None
        print(f"{'OK' if ok else 'ABSENT'} : Python {name}")
        if not ok:
            missing.append(name)
    for name in ("ballDetection_yolo26x_best.pt", "ballDetection_yolo11x_best.pt"):
        path = BACKEND / "MachineLearning" / "models" / name
        ok = path.is_file() and path.stat().st_size > 0
        print(f"{'OK' if ok else 'ABSENT'} : {path}")
        if not ok:
            missing.append(str(path))
    status = ANALYSIS / "GameStatusDetection" / "Models"
    checkpoint = status / "Custom" / "checkpoint"
    if not checkpoint.exists():
        checkpoint = status / "VolleyballAnalytics" / "3-states" / "checkpoint"
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        path = checkpoint / name
        ok = path.is_file() and path.stat().st_size > 0
        print(f"{'OK' if ok else 'ABSENT'} : {path}")
        if not ok:
            missing.append(str(path))
    if not (ANALYSIS / "ActionDetection" / "action_detector.pt").is_file():
        print("INFO : action_detector.pt absent ; classification heuristique du projet.")
    print("INFO : yolo26x.pt et yolo26m-reid.onnx peuvent etre telecharges au premier lancement.")
    return not missing


def video_info(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Video illisible : {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frames <= 0:
            raise RuntimeError(f"Duree/FPS non disponibles : {path}")
        return {"fps": fps, "frames": int(frames), "duration_s": frames / fps}
    finally:
        cap.release()


def extract(source, start, duration):
    import imageio_ffmpeg

    info = video_info(source)
    if start + duration > info["duration_s"] + 1 / info["fps"]:
        raise ValueError(f"Extrait hors video (duree source : {info['duration_s']:.2f} s).")
    # Nouveau dossier par essai pour ne pas melanger calibrations et resultats.
    run = HERE / "outputs" / datetime.now().strftime("chevry_%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True)
    clip = run / "chevry_raw.mp4"
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-n", "-ss", str(start), "-i", str(source), "-t", str(duration),
        "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-crf", "18",
        "-preset", "fast", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-movflags", "+faststart", str(clip),
    ], check=True)
    actual = video_info(clip)
    if abs(actual["duration_s"] - duration) > max(0.1, 2 / actual["fps"]):
        raise RuntimeError(f"Duree extraite inattendue : {actual['duration_s']:.3f} s")
    (run / "extraction.json").write_text(json.dumps({
        "source": str(source), "start_s": start, "requested_duration_s": duration,
        "clip": actual, "note": "Extrait continu, temps morts inclus, sans ajout d'annotation.",
    }, indent=2), encoding="utf-8")
    print(f"Extrait non annote : {clip}", flush=True)
    return clip, run


def analyse(clip, run):
    sys.path[:0] = [str(ANALYSIS), str(BACKEND)]
    # Le projet utilise aussi des noms de modeles relatifs au dossier courant.
    os.chdir(BACKEND)
    stages = [
        ("CourtDetection.court", "detectCourt", "court.json"),
        ("GameStatusDetection.gameStatusDetection", "detectGameStatus", "game_status.json"),
        ("PlayerDetection.tracker", "trackplayers_offline", "player_positions.json"),
        ("BallDetection.ballDetection", "detectBall", "ball_speed.json"),
        ("ActionDetection.actionDetection", "detectActions", "actions.json"),
    ]
    # Importer avant d'ouvrir la calibration pour reperer les incompatibilites.
    functions = [(getattr(importlib.import_module(m), f), out) for m, f, out in stages]
    from ultralytics.trackers.utils.reid import ReID  # noqa: F401
    from PostProcessing.consolidate import consolidateStats
    from PostProcessing.generate_dashboard import generateDashboard
    from PostProcessing.renderVideo import renderAnnotatedVideo

    result = run / "analysis"
    result.mkdir()
    print("Calibration : ajuster les points terrain/filet, puis Entree. Echap annule.", flush=True)
    for function, expected in functions:
        print(f"\nEtape : {function.__name__}", flush=True)
        function(str(clip), str(result))
        if not (result / expected).is_file():
            raise RuntimeError(f"{expected} absent apres {function.__name__} (annulation ou echec).")
        if expected == "game_status.json":
            if not json.loads((result / expected).read_text())["rallies"]:
                raise RuntimeError("Aucun echange detecte. Verifier l'extrait et le checkpoint.")
    consolidateStats(str(result))
    generateDashboard(str(result))
    renderAnnotatedVideo(str(clip), str(result))
    if not (result / "analysis.mp4").is_file():
        raise RuntimeError("Le rendu analysis.mp4 est absent.")
    video_info(result / "analysis.mp4")
    print(f"\nResultats : {result}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "extract", "run"), default="run")
    parser.add_argument("--start", type=float, default=60, help="Debut en secondes (defaut : 60).")
    parser.add_argument("--duration", type=float, default=120, help="Duree en secondes (defaut : 120).")
    parser.add_argument("--source", type=Path, help="MP4 source ; sinon recherche Chevry dans data.")
    args = parser.parse_args()
    if args.mode == "check":
        return 0 if preflight() else 1
    if not math.isfinite(args.start) or not math.isfinite(args.duration) or args.start < 0 or args.duration <= 0:
        parser.error("--start doit etre >= 0 et --duration > 0, valeurs finies.")
    candidates = sorted(p for p in (ROOT / "data").rglob("*")
                        if p.is_file() and p.suffix.lower() == ".mp4" and "chevry" in p.name.lower())
    if args.source:
        source = args.source.resolve()
    elif len(candidates) == 1:
        source = candidates[0]
    else:
        parser.error(f"{len(candidates)} videos Chevry trouvees : preciser --source.")
    if not source.is_file():
        parser.error(f"Video absente : {source}")
    print(f"Source : {source}", flush=True)
    if args.mode == "run" and not preflight():
        print("\nPrerequis manquants : voir test-SOA/README.md. Extraction seule : --mode extract.")
        return 1
    clip, run = extract(source, args.start, args.duration)
    if args.mode == "run":
        analyse(clip, run)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ImportError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        sys.exit(1)
