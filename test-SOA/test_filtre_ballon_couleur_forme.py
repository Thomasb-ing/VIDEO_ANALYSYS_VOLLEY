"""Compare les filtres jaune/bleu sur 40 images detectees par VballNet.

Filtres HSV uniquement : aucun critere de forme n'est applique a ce stade.
Les images sont lues dans la video NON annotee pour ne pas filtrer les overlays.
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def save(path, frame):
    # imencode/tofile supporte les chemins Windows contenant des accents.
    ok, data = cv2.imencode(path.suffix, frame)
    if not ok:
        raise RuntimeError(f"Encodage impossible : {path}")
    data.tofile(str(path))


def label(image, text):
    result = cv2.copyMakeBorder(image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    cv2.putText(result, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Dossier VballNet avec summary.json et detections.csv.")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--s-min", type=int, default=100, help="Saturation HSV minimale (0..255).")
    parser.add_argument("--v-min", type=int, default=80, help="Luminosite HSV minimale (0..255).")
    parser.add_argument("--crop-size", type=int, default=96, help="Cote du zoom source, en pixels.")
    args = parser.parse_args()
    if args.count <= 0 or args.crop_size <= 0 or not 0 <= args.s_min <= 255 or not 0 <= args.v_min <= 255:
        parser.error("count/crop-size doivent etre positifs et s-min/v-min entre 0 et 255")
    run = args.run.resolve() if args.run else None
    if run is None:
        candidates = []
        for p in (HERE / "outputs").glob("vball_standard_*/summary.json"):
            meta = json.loads(p.read_text(encoding="utf-8"))
            if meta.get("status") == "complete" and meta.get("processed_duration_s", 0) >= 120:
                candidates.append(p)
        if not candidates:
            parser.error("Aucun essai standard de 120s trouve ; preciser --run.")
        run = max(candidates, key=lambda p: p.stat().st_mtime_ns).parent
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    source = Path(summary["video"])
    with (run / "detections.csv").open(encoding="utf-8", newline="") as f:
        detections = [r for r in csv.DictReader(f) if r["detected"] == "1" and r.get("warmup", "0") == "0"]
    detections.sort(key=lambda r: int(r["frame"]))
    if len(detections) < args.count:
        raise ValueError(f"Seulement {len(detections)} detections disponibles pour {args.count} images.")
    # Echantillon deterministe reparti dans la liste des detections, sans selection visuelle.
    indices = np.linspace(0, len(detections) - 1, args.count, dtype=int)
    selected = [detections[i] for i in indices]
    targets = {int(r["frame"]): r for r in selected}
    if len(targets) != args.count:
        raise ValueError("Indices d'images dupliques dans le CSV.")
    output = HERE / "outputs" / datetime.now().strftime("filtres_couleur_%Y%m%d_%H%M%S_%f")
    images = output / "images"
    images.mkdir(parents=True)
    # OpenCV : teinte H sur 0..179, S et V sur 0..255.
    ranges = {"jaune": (20, 35), "bleu": (95, 130)}
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Video illisible : {source}")
    records, crops, previews = [], [], []
    try:
        # Lecture sequentielle : garantit les indices exacts sans seek approximatif.
        for index in range(max(targets) + 1):
            if not cap.grab():
                raise RuntimeError(f"Video interrompue a l'image {index}")
            if index not in targets:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                raise RuntimeError(f"Image {index} illisible")
            row = targets[index]
            x, y = int(row["x_px"]), int(row["y_px"])
            height, width = frame.shape[:2]
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Detection hors image : {index}")
            prefix = f"{len(records)+1:02d}_frame_{index:06d}"
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            masks = {name: cv2.inRange(hsv, (low, args.s_min, args.v_min), (high, 255, 255))
                     for name, (low, high) in ranges.items()}
            filtered = {name: cv2.bitwise_and(frame, frame, mask=mask) for name, mask in masks.items()}
            # Les versions filtrees gardent les couleurs source ; noir ailleurs.
            save(images / f"{prefix}_original.jpg", frame)
            for name in ranges:
                save(images / f"{prefix}_{name}.png", filtered[name])
                save(images / f"{prefix}_masque_{name}.png", masks[name])
            left = max(0, min(x - args.crop_size // 2, width - args.crop_size))
            top = max(0, min(y - args.crop_size // 2, height - args.crop_size))
            right, bottom = min(width, left + args.crop_size), min(height, top + args.crop_size)
            record = {"sample": len(records)+1, **row, "prefix": prefix,
                      "crop_left": left, "crop_top": top, "crop_right": right, "crop_bottom": bottom}
            for name, mask in masks.items():
                record[f"{name}_fraction_crop"] = float(np.count_nonzero(mask[top:bottom, left:right]) / ((bottom-top)*(right-left)))
            panels = []
            for name, image in [("original", frame), *filtered.items()]:
                crop = image[top:bottom, left:right]
                crop = cv2.resize(crop, (240, 240), interpolation=cv2.INTER_NEAREST)
                # Croix blanche ajoutee APRES filtrage, uniquement sur les comparaisons.
                point = (int((x-left)*240/(right-left)), int((y-top)*240/(bottom-top)))
                cv2.drawMarker(crop, point, (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
                panels.append(label(crop, f"{record['sample']:02d} | {name} | f{index}"))
            comparison = np.hstack(panels)
            save(images / f"{prefix}_zoom.jpg", comparison)
            crops.append(comparison)
            if len(previews) < 4:
                whole = []
                for name, image in [("original", frame), *filtered.items()]:
                    mini = cv2.resize(image, (640, round(height*640/width)))
                    cv2.rectangle(mini, (round(left*640/width), round(top*640/width)),
                                  (round(right*640/width), round(bottom*640/width)), (255, 255, 255), 1)
                    whole.append(label(mini, f"{name} | image {index} | t={float(row['time_clip_s']):.2f}s"))
                preview_name = f"apercu_{record['sample']:02d}.jpg"
                save(output / preview_name, np.hstack(whole))
                previews.append(preview_name)
            records.append(record)
    finally:
        cap.release()
    sheets = []
    for offset in range(0, len(crops), 10):
        name = f"planche_zooms_{offset//10+1:02d}.jpg"
        save(output / name, np.vstack(crops[offset:offset+10]))
        sheets.append(name)
    with (output / "selection.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (output / "parametres.json").write_text(json.dumps({
        "source_video": str(source), "vball_run": str(run), "count": len(records),
        "hsv_ranges_opencv": ranges, "s_min": args.s_min, "v_min": args.v_min,
        "crop_size": args.crop_size, "selection": "Uniform indices among detected frames, excluding warmup",
        "notes": ["Color only; no shape filter or morphological cleanup.",
                  "Vball detections are proposals, not verified ground truth.",
                  "White crosses/rectangles appear only in comparison previews, after filtering."],
    }, indent=2), encoding="utf-8")
    cards = []
    for r in records:
        prefix = r["prefix"]
        links = ' | '.join(f'<a href="images/{prefix}_{suffix}">{title}</a>' for suffix, title in [
            ("original.jpg", "Original"), ("jaune.png", "Jaune"), ("bleu.png", "Bleu"),
            ("masque_jaune.png", "Masque jaune"), ("masque_bleu.png", "Masque bleu")])
        cards.append(f'<section><h2>Image {r["sample"]} — frame {r["frame"]} — extrait {float(r["time_clip_s"]):.2f} s</h2>'
                     f'<p>{links}</p><img loading="lazy" src="images/{prefix}_zoom.jpg" alt="Original, jaune, bleu"></section>')
    page = ('<!doctype html><html lang="fr"><meta charset="utf-8"><title>Filtres ballon</title>'
            '<style>body{font-family:Arial;background:#181818;color:#eee;max-width:1200px;margin:24px auto;padding:16px}'
            'a{color:#8ed0ff}img{max-width:100%;height:auto}section{border-top:1px solid #555;padding:16px 0}</style>'
            '<h1>Ballon : original / jaune / bleu</h1>'
            f'<p>{len(records)} images signalées par VballNet. Les détections ne sont pas validées manuellement.</p>'
            '<p>Filtrage HSV sur les images brutes : jaune H=20–35, bleu H=95–130 (échelle OpenCV 0–179). '
            f'S ≥ {args.s_min}, V ≥ {args.v_min}. Couleurs source conservées, autres pixels noirs. Aucun filtre de forme.</p>'
            '<p>La croix indique la position VballNet. Elle est ajoutée après filtrage. Les liens donnent les images sans repères.</p>'
            '<h2>Vues complètes</h2>' + ''.join(f'<a href="{p}"><img src="{p}" alt="Comparaison pleine image"></a>' for p in previews)
            + ''.join(cards) + '</html>')
    (output / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(records)} images exportees : {output}")
    print(f"Galerie : {output / 'index.html'}")


if __name__ == "__main__":
    main()
