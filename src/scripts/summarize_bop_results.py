"""
Collect the local bop_toolkit scores of every evaluated BOP dataset into a table aligned
with Table 1 of the CNOS paper (segmentation AP, BOP'19 test targets).

usage: python -m src.scripts.summarize_bop_results [--root datasets/bop23_challenge] [--method cnosv2] [--markdown out.md]
"""
import argparse
import json
import os.path as osp

PAPER_ORDER = ["lmo", "tless", "tudl", "icbin", "itodd", "hb", "ycbv"]
PAPER_NAMES = {"lmo": "LM-O", "tless": "T-LESS", "tudl": "TUD-L", "icbin": "IC-BIN",
               "itodd": "ITODD", "hb": "HB", "ycbv": "YCB-V"}
# Table 1 of the CNOS paper (ICCV-W 2023): segmentation AP, model-based, BOP'19 targets
PAPER = {
    "CNOS (SAM) + Pyrender":    {"lmo": 33.3, "tless": 38.3, "tudl": 35.8, "icbin": 27.2, "itodd": 14.8, "hb": 45.9, "ycbv": 57.6},
    "CNOS (SAM) + BlenderProc": {"lmo": 39.6, "tless": 39.7, "tudl": 39.1, "icbin": 28.4, "itodd": 28.2, "hb": 48.0, "ycbv": 59.5},
    "CNOS (FastSAM) + BlenderProc": {"lmo": 39.7, "tless": 37.4, "tudl": 48.0, "icbin": 27.0, "itodd": 25.4, "hb": 51.1, "ycbv": 59.9},
}
SPLIT_SUFFIX = {"tless": "-primesense", "hb": "-primesense"}


def load_scores(root, method, ds):
    d = osp.join(root, "results", "bop_eval", f"{method}_{ds}-test{SPLIT_SUFFIX.get(ds, '')}")
    out = {}
    for t in ["segm", "bbox"]:
        p = osp.join(d, f"scores_bop22_coco_{t}.json")
        if osp.exists(p):
            out[t] = json.load(open(p))
    return out


def fmt(x):
    return "-" if x is None else f"{x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./datasets/bop23_challenge")
    ap.add_argument("--method", default="cnosv2")
    ap.add_argument("--markdown", default=None, help="also write the tables to this markdown file")
    args = ap.parse_args()

    scores = {ds: load_scores(args.root, args.method, ds) for ds in PAPER_ORDER}
    pred_dir = osp.join(args.root, "results", "cnos_exps")

    def row(name, vals, mean_over=None):
        cells = [fmt(vals.get(ds)) for ds in PAPER_ORDER]
        present = [vals[ds] for ds in (mean_over or PAPER_ORDER) if vals.get(ds) is not None]
        mean = sum(present) / len(present) if present else None
        return f"| {name} | " + " | ".join(cells) + f" | {fmt(mean)} |"

    header = "| Method | " + " | ".join(PAPER_NAMES[d] for d in PAPER_ORDER) + " | Mean |\n|---|" + "---|" * (len(PAPER_ORDER) + 1)
    ours_segm = {ds: s["segm"]["AP"] * 100 for ds, s in scores.items() if "segm" in s}
    ours_bbox = {ds: s["bbox"]["AP"] * 100 for ds, s in scores.items() if "bbox" in s}
    evaluated = [ds for ds in PAPER_ORDER if ds in ours_segm]
    missing = [ds for ds in PAPER_ORDER if ds not in ours_segm]

    lines = []
    lines.append("### 2D segmentation AP (BOP'19 test targets, pyrender templates, 42 views)")
    lines.append("")
    lines.append(header)
    for name, vals in PAPER.items():
        lines.append(row(name + " [paper]", vals))
        lines.append(row(name + f" [paper, mean over {len(evaluated)} evaluated sets]", vals, mean_over=evaluated))
    lines.append(row("**CNOSv2 (SAM2.1-L + DINOv3-L) + Pyrender**", ours_segm))
    lines.append("")
    lines.append("### 2D detection AP (amodal bbox), ours only")
    lines.append("")
    lines.append(header)
    lines.append(row("CNOSv2 (SAM2.1-L + DINOv3-L) + Pyrender", ours_bbox))
    lines.append("")
    lines.append("### Per-dataset details (ours)")
    lines.append("")
    lines.append("| Dataset | images | detections | segm AP | AP50 | AP75 | AR100 | bbox AP | time / image |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for ds in PAPER_ORDER:
        s = scores[ds]
        pred = osp.join(pred_dir, f"CustomSam2AutomaticMaskGenerator_template_pyrender0_aggavg_5_{ds}.json")
        n_img = n_det = "-"
        if osp.exists(pred):
            d = json.load(open(pred))
            n_det = len(d); n_img = len({(x["scene_id"], x["image_id"]) for x in d})
        if "segm" in s:
            g, b = s["segm"], s.get("bbox", {})
            lines.append(f"| {PAPER_NAMES[ds]} | {n_img} | {n_det} | {g['AP']*100:.1f} | {g['AP50']*100:.1f} | {g['AP75']*100:.1f} | {g['AR100']*100:.1f} | {fmt(b.get('AP', None) and b['AP']*100)} | {g['average_time_per_image']:.2f} s |")
        else:
            note = "no test GT in the BOP release, score on bop.felk.cvut.cz" if osp.exists(pred) else "not run"
            lines.append(f"| {PAPER_NAMES[ds]} | {n_img} | {n_det} | - | - | - | - | - | {note} |")
    if missing:
        lines.append("")
        lines.append(f"Not locally evaluable: {', '.join(PAPER_NAMES[d] for d in missing)} (their BOP'19 test archives ship no scene_gt.json); "
                     f"the paper rows' last column also shows the mean over the {len(evaluated)} evaluated datasets for a like-for-like comparison.")
    text = "\n".join(lines)
    print(text)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(text + "\n")
        print(f"\nwritten to {args.markdown}")


if __name__ == "__main__":
    main()
