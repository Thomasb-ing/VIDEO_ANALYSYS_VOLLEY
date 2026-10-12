"""Teste les poids ONNX VballNet sur un extrait Chevry existant.

Pre/post-traitement adapte de vball-net/src/inference_onnx.py (depot voisin).
CSV via la bibliotheque standard : pandas/TensorFlow ne sont pas necessaires.
"""

import argparse
from collections import deque
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
MODELS = {
    "fast": "VballNetFastV1_155_h288_w512.onnx",
    "standard": "VballNetV1_150_h288_w512.onnx",
}


def ball_position(heatmap, threshold, cv2, np):
    """Comme le depot : centroide du plus grand contour au-dessus du seuil."""
    binary = (heatmap > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            _, _, w, h = cv2.boundingRect(contour)
            return (int(moments["m10"] / moments["m00"]),
                    int(moments["m01"] / moments["m00"]), w, h)
    return None


def run(args):
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort

    if args.video:
        video = args.video.resolve()
    else:
        candidates = list((HERE / "outputs").glob("chevry_*/chevry_raw.mp4"))
        if not candidates:
            raise ValueError("Aucun extrait Chevry. Lancer test_chevry.py --mode extract ou utiliser --video.")
        video = max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)))
    model_path = HERE / "models" / MODELS[args.model]
    if not video.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"Video ou modele absent : {video} ; {model_path}")
    metadata_path = video.with_name("extraction.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    match_start = metadata.get("start_s")
    if match_start is not None:
        match_start = float(match_start)
        if not math.isfinite(match_start) or match_start < 0:
            raise ValueError("start_s invalide dans extraction.json")

    onnx.checker.check_model(onnx.load(str(model_path), load_external_data=False))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    if input_info.shape[1:] != [9, 288, 512] or input_info.type != "tensor(float)":
        raise ValueError(f"Entree modele incompatible : {input_info.shape}, {input_info.type}")
    cap = cv2.VideoCapture(str(video))
    writer = None
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Video illisible : {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or min(total, width, height) <= 0:
            raise ValueError("Metadonnees video invalides")
        limit = max(1, round(args.duration * fps))
        if limit > total:
            raise ValueError(f"Duree demandee {args.duration}s > extrait disponible {total / fps:.3f}s")
        output = HERE / "outputs" / datetime.now().strftime(f"vball_{args.model}_%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True)
        print(f"Video : {video}\nModele : {model_path.name}\nSorties : {output}", flush=True)
        writer = cv2.VideoWriter(str(output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Impossible de creer annotated.mp4 (codec mp4v).")
        frames = deque(maxlen=3)
        trail = deque(maxlen=8)
        detected = 0
        started = time.perf_counter()
        columns = ["frame", "time_clip_s", "time_match_s", "detected", "x_px", "y_px",
                   "w_px", "h_px", "heatmap_max", "warmup"]
        with (output / "detections.csv").open("w", newline="", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=columns)
            csv_writer.writeheader()
            for index in range(limit):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Lecture interrompue a l'image {index}/{limit} ; sorties partielles.")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed = cv2.resize(rgb, (512, 288)).astype(np.float32) / 255.0
                frames.append(processed)
                # Meme initialisation que le depot : premiere image repetee.
                while len(frames) < 3:
                    frames.append(processed)
                tensor = np.concatenate(frames, axis=2)[None].transpose(0, 3, 1, 2)
                prediction = session.run(None, {input_info.name: tensor})[0]
                if prediction.shape != (1, 3, 288, 512) or not np.isfinite(prediction).all():
                    raise RuntimeError("Sortie du modele invalide")
                heatmap = prediction[0, 2]  # La troisieme carte correspond a l'image courante.
                position = ball_position(heatmap, args.threshold, cv2, np)
                coords = [None] * 4
                if position is not None:
                    coords = [int(value * scale) for value, scale in
                              zip(position, (width / 512, height / 288, width / 512, height / 288))]
                    detected += 1
                # Une entree par image, y compris les absences, pour ne pas figer la trace.
                trail.append(tuple(coords[:2]) if position else None)
                csv_writer.writerow(dict(zip(columns, [
                    index, index / fps, match_start + index / fps if match_start is not None else None,
                    int(position is not None), *coords, float(heatmap.max()), int(index < 2),
                ])))
                for point in list(trail)[:-1]:
                    if point is not None:
                        cv2.circle(frame, point, 5, (255, 0, 0), -1)
                if trail[-1] is not None:
                    cv2.circle(frame, trail[-1], 5, (0, 0, 255), -1)
                cv2.putText(frame, f"{args.model} | {index / fps:.2f}s | detected={int(position is not None)}",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                writer.write(frame)
                if index == 0 or (index + 1) % 100 == 0 or index + 1 == limit:
                    print(f"{index + 1}/{limit} images", flush=True)
        writer.release()
        writer = None
        elapsed = time.perf_counter() - started
        summary = {
            "status": "complete", "video": str(video), "source_match": metadata.get("source"),
            "match_start_s": match_start, "model": str(model_path),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "providers": session.get_providers(), "onnxruntime_version": ort.__version__,
            "threshold": args.threshold, "requested_duration_s": args.duration,
            "processed_duration_s": limit / fps, "fps": fps, "frames": limit,
            "detected_frames": detected, "detection_fraction": detected / limit,
            "processing_seconds": elapsed, "processing_fps": limit / elapsed,
            "notes": ["Detection fraction is not accuracy; heatmap_max is not calibrated confidence.",
                      "First two frames use repeated initial frame (warmup=1).",
                      "Coordinates are original-image pixels; no smoothing/interpolation. Output video has no audio.",
                      "Pre/postprocessing adapted from vball-net/src/inference_onnx.py."],
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Termine : {limit} images en {elapsed:.2f}s. Resultats : {output}", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="fast")
    parser.add_argument("--duration", type=float, default=10, help="Secondes depuis le debut de l'extrait (defaut 10).")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--video", type=Path, help="Sinon, utilise le dernier extrait outputs/chevry_*/chevry_raw.mp4.")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration doit etre finie et > 0")
    if not math.isfinite(args.threshold) or not 0 < args.threshold < 1:
        parser.error("--threshold doit etre entre 0 et 1, bornes exclues")
    run(args)


if __name__ == "__main__":
    try:
        main()
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        sys.exit(1)
