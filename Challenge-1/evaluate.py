import os
import numpy as np
import cv2
from glob import glob
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_score(img_a, img_b):
    p = peak_signal_noise_ratio(img_a, img_b, data_range=255)
    s = structural_similarity(img_a, img_b, channel_axis=2, data_range=255)
    return p, s, p * s


def load_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Immagine non trovata: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def evaluate(new_dir, best_dir, min_psnr=20.0, min_ssim=0.8):
    """
    Confronta le immagini in new_dir contro quelle in best_dir (riferimento).
    Un nuovo risultato è "migliore" se ha score più alto del best precedente,
    ovvero se è più simile al riferimento best.

    Parameters:
    - new_dir   : cartella con le nuove immagini restaurate
    - best_dir  : cartella con le immagini best attuali (riferimento)
    - min_psnr  : soglia minima PSNR per considerare un'immagine accettabile
    - min_ssim  : soglia minima SSIM per considerare un'immagine accettabile
    """

    best_paths = sorted(glob(os.path.join(best_dir, '*.jpg')) +
                        glob(os.path.join(best_dir, '*.png')))
    if not best_paths:
        print(f"[ERRORE] Nessuna immagine trovata in: {best_dir}")
        return

    results = []

    print(f"{'File':<20} {'PSNR':>10} {'SSIM':>10} {'Score':>10}  {'PSNR ok':>8} {'SSIM ok':>8}")
    print("-" * 75)

    for path_best in best_paths:
        filename = os.path.basename(path_best)
        path_new = os.path.join(new_dir, filename)

        if not os.path.exists(path_new):
            print(f"[WARN] {filename} non trovato in new_dir, skip")
            continue

        best = load_rgb(path_best)
        new  = load_rgb(path_new)

        if best.shape != new.shape:
            print(f"[WARN] {filename} shape diversa ({new.shape} vs {best.shape}), skip")
            continue

        p, s, sc = compute_score(new, best)

        psnr_ok = "✓" if p >= min_psnr else "✗"
        ssim_ok = "✓" if s >= min_ssim else "✗"

        results.append({'filename': filename, 'psnr': p, 'ssim': s, 'score': sc})

        print(f"{filename:<20} {p:>10.3f} {s:>10.4f} {sc:>10.4f}  {psnr_ok:>8} {ssim_ok:>8}")

    if not results:
        print("Nessun risultato calcolato.")
        return

    mean_p  = np.mean([r['psnr']  for r in results])
    mean_s  = np.mean([r['ssim']  for r in results])
    mean_sc = np.mean([r['score'] for r in results])
    n_psnr_ok = sum(1 for r in results if r['psnr'] >= min_psnr)
    n_ssim_ok = sum(1 for r in results if r['ssim'] >= min_ssim)
    n_total   = len(results)

    print("-" * 75)
    print(f"\nSOMMARIO ({n_total} immagini)")
    print(f"  PSNR medio : {mean_p:.3f}  (>={min_psnr}: {n_psnr_ok}/{n_total})")
    print(f"  SSIM medio : {mean_s:.4f}  (>={min_ssim}: {n_ssim_ok}/{n_total})")
    print(f"  Score medio: {mean_sc:.4f}")
    print(f"\n  Interpretazione: score vicino a 0 = new diverge da best, score alto = new è simile a best")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Confronta nuove immagini restaurate contro il best attuale.")
    parser.add_argument("new_dir",  help="Cartella con le nuove immagini restaurate")
    parser.add_argument("best_dir", help="Cartella con le immagini best (riferimento)")
    parser.add_argument("--min_psnr", type=float, default=20.0)
    parser.add_argument("--min_ssim", type=float, default=0.8)
    args = parser.parse_args()

    evaluate(args.new_dir, args.best_dir, args.min_psnr, args.min_ssim)