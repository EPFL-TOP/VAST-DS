"""Profile-based somite detection.

Premise: zebrafish somites are periodic along the AP (head→tail) axis. After
the orientation classifier canonicalises the fish (head left, tail right),
the periodicity sits along the image x-axis, so a 1-D peak-finding problem
gives us per-somite x positions, count and body length essentially for free.

For each y-strip across the spine ROI we compute an intensity profile, find
peaks (somite boundaries), then cluster consensus peaks across strips. The
*upper* and *lower* half of the spine ROI each get their own confidence
score — that's what catches asymmetric defects (a somite visible at the top
of the spine but missing at the bottom).

Public surface:
    analyze_image(image_2d, **opts) -> dict
        keys:
            somites       : list of per-somite dicts (see below)
            body_length   : float, pixel distance between first & last somite
            spine_y_center: int
            spine_dy      : int
            mean_profile  : numpy array, smoothed mean profile (for plotting)
            n_strips      : int, number of y-strips used

Each somite dict (also the shape that lives in
DestWellPropertiesPredicted.per_somite_data):
    index             : 0-based, AP order
    centroid_x        : peak x
    centroid_y        : spine centre (same y for all somites — a single fish)
    bbox              : [x0, y0, x1, y1] — the per-somite ROI, suitable for
                        cropping into a tile for a future defect classifier
    ap_position       : normalised 0..1 along the AP axis
    confidence        : fraction of all strips that found this peak
    upper_confidence  : fraction of UPPER-half strips that found it
    lower_confidence  : fraction of LOWER-half strips that found it
    intensity         : mean peak height across strips
    severity          : 0=healthy, 1=dim, 2=asymmetric, 3=very weak
    severity_reason   : short string for the UI
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, medfilt


# Sensible defaults for ~512×512 corrected YFP fish images.
DEFAULTS = dict(
    n_strips=20,            # how many y-positions to scan
    strip_thickness=3,      # half-height of each strip (rows averaged)
    smoothing_sigma=3.0,    # gaussian smoothing of the 1-D profile
    detrend_sigma=50.0,     # window for the high-pass baseline subtraction;
                            # set to 0 to disable. Should be a few × somite
                            # spacing so the per-somite oscillation is
                            # preserved while the slow head→tail intensity
                            # gradient is removed.
    peak_prominence=0.015,  # in normalised intensity units (0..1) — applied
                            # to the DETRENDED profile, so this is the
                            # somite-oscillation amplitude, not the raw
                            # signal amplitude
    peak_distance=15,       # min px between peaks ≈ shortest plausible somite
    cluster_window=12,      # px window to declare two strip-peaks the same somite
    min_confidence=0.30,    # discard candidates seen in fewer strips than this
    spine_frac=0.5,         # FWHM-like fraction used to derive spine half-height
    straighten=True,        # follow the spine curve before scanning strips
    centerline_smooth=21,   # median-filter window for centerline (odd integer)
    centerline_poly=3,      # degree of polynomial fit over the centerline
)


# ---------------------------------------------------------------------------
# Spine centerline & image straightening
# ---------------------------------------------------------------------------
def find_spine_centerline(image: np.ndarray,
                          vert_sigma: float = 3.0,
                          smooth_window: int = DEFAULTS['centerline_smooth'],
                          poly_deg: int = DEFAULTS['centerline_poly'],
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_spine, valid_mask) — y position of the spine per x column.

    For each x column we find the y of maximum (vertically smoothed)
    intensity, then median-filter the resulting curve to suppress hot
    pixels and fit a low-order polynomial to globally smooth it. The
    polynomial is fit only on columns bright enough to contain the fish;
    the rest of the columns are extrapolated by the polynomial, so the
    output is defined for every x.
    """
    h, w = image.shape
    smoothed = gaussian_filter1d(image, sigma=vert_sigma, axis=0)
    y_peak = smoothed.argmax(axis=0).astype(np.float32)

    # Mask out columns where the fish almost certainly isn't (dark cols).
    col_max = image.max(axis=0)
    threshold = max(0.1 * col_max.max(), 0.05)
    valid = col_max > threshold
    if valid.sum() < 10:
        valid = col_max > col_max.mean() * 0.5

    # Median-smooth to reject outliers
    if smooth_window % 2 == 0:
        smooth_window += 1
    y_smooth = medfilt(y_peak, kernel_size=smooth_window)

    # Polynomial fit on valid columns; evaluated everywhere
    if valid.sum() > poly_deg + 2:
        xs = np.arange(w, dtype=np.float32)[valid]
        ys = y_smooth[valid]
        coeffs = np.polyfit(xs, ys, deg=poly_deg)
        y_smooth = np.polyval(coeffs, np.arange(w, dtype=np.float32))
    # Clip to image bounds
    y_smooth = np.clip(y_smooth, 0, h - 1).astype(np.float32)
    return y_smooth, valid


def straighten_image(image: np.ndarray,
                     y_spine: np.ndarray,
                     target_y: Optional[int] = None,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Vertically shift each column so that y_spine(x) → target_y.

    After this transform the spine is horizontal at row `target_y` and
    the existing horizontal-strip analysis recovers chevron peaks that
    would otherwise drift out of any single strip when the fish is curved.
    Returns the straightened image AND the per-column shift array.
    """
    h, w = image.shape
    if target_y is None:
        target_y = h // 2
    shifts = (target_y - y_spine).astype(np.int32)
    out = np.zeros_like(image)
    for x in range(w):
        s = int(shifts[x])
        if s > 0:
            out[s:, x] = image[:h - s, x]
        elif s < 0:
            out[:h + s, x] = image[-s:, x]
        else:
            out[:, x] = image[:, x]
    return out, shifts


# ---------------------------------------------------------------------------
# Spine ROI
# ---------------------------------------------------------------------------
def find_spine_roi(image: np.ndarray, frac: float = DEFAULTS['spine_frac']
                   ) -> Tuple[int, int]:
    """Return (y_center, dy_half_height) for the band containing the spine.

    Uses the row-mean projection — the strongest row corresponds to the
    midline; the spine extent is everything above `frac * (max - min) + min`.
    """
    h = image.shape[0]
    row_mean = image.mean(axis=1)
    y_center = int(np.argmax(row_mean))
    threshold = row_mean.min() + frac * (row_mean.max() - row_mean.min())
    above = np.where(row_mean >= threshold)[0]
    if above.size == 0:
        return y_center, max(h // 6, 10)
    y_min, y_max = int(above.min()), int(above.max())
    dy = max((y_max - y_min) // 2, 10)
    return (y_min + y_max) // 2, dy


# ---------------------------------------------------------------------------
# Per-strip profiles + peak finding
# ---------------------------------------------------------------------------
def _strip_profile(image: np.ndarray, y: int, thickness: int,
                   sigma: float) -> np.ndarray:
    """Average a few rows around `y`, smooth, return the 1-D x-profile."""
    h = image.shape[0]
    y0 = max(0, y - thickness)
    y1 = min(h, y + thickness + 1)
    strip = image[y0:y1, :].mean(axis=0)
    if sigma > 0:
        strip = gaussian_filter1d(strip, sigma=sigma)
    return strip


def _peaks_for_strip(profile: np.ndarray, prominence: float,
                     distance: int) -> np.ndarray:
    peaks, _ = find_peaks(profile, prominence=prominence, distance=distance)
    return peaks


# ---------------------------------------------------------------------------
# Cluster strip-level peaks into per-somite detections
# ---------------------------------------------------------------------------
def _cluster_peaks(strip_peaks: List[Tuple[int, int, float]],
                   window: int) -> List[List[Tuple[int, int, float]]]:
    """Group strip-level peaks across y into per-somite clusters.

    `strip_peaks` is a list of (strip_idx, x, intensity). We sort by x and
    open a new cluster every time the x gap exceeds `window`.
    """
    if not strip_peaks:
        return []
    s = sorted(strip_peaks, key=lambda t: t[1])
    clusters: List[List[Tuple[int, int, float]]] = []
    cur: List[Tuple[int, int, float]] = [s[0]]
    prev_x = s[0][1]
    for t in s[1:]:
        if t[1] - prev_x <= window:
            cur.append(t)
        else:
            clusters.append(cur)
            cur = [t]
        prev_x = t[1]
    if cur:
        clusters.append(cur)
    return clusters


def _classify_severity(conf: float, upper: float, lower: float,
                       intensity: float, median_intensity: float) -> Tuple[int, str]:
    """Map (confidence, asymmetry, intensity) → severity 0-3 + reason string."""
    asym = abs(upper - lower)
    if conf < 0.60:
        return 3, f'weak detection (conf={conf:.2f})'
    if asym > 0.35:
        return 2, (f'asymmetric: upper={upper:.2f} vs lower={lower:.2f}')
    if median_intensity > 0 and intensity < 0.55 * median_intensity:
        return 1, (f'dim ({intensity:.2f} vs median {median_intensity:.2f})')
    return 0, 'healthy'


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def analyze_image(image: np.ndarray,
                  n_strips: int = DEFAULTS['n_strips'],
                  strip_thickness: int = DEFAULTS['strip_thickness'],
                  smoothing_sigma: float = DEFAULTS['smoothing_sigma'],
                  detrend_sigma: float = DEFAULTS['detrend_sigma'],
                  peak_prominence: float = DEFAULTS['peak_prominence'],
                  peak_distance: int = DEFAULTS['peak_distance'],
                  cluster_window: int = DEFAULTS['cluster_window'],
                  min_confidence: float = DEFAULTS['min_confidence'],
                  straighten: bool = DEFAULTS['straighten'],
                  ) -> Dict:
    """Run the profile-based detector on a single 2-D image.

    `image` should already be normalised (e.g. through `_common.preprocess_image`
    so values are in [0, 1]). When `straighten=True` (the default) the
    spine is first fitted as a polynomial in (x, y) and the image is
    vertically warped so the fish lies along a horizontal line — without
    this step, the AP-axis-aligned strip scan misses peaks on the curved
    half of the fish.

    Returns a dict with:
        somites             : list of per-somite dicts (see module docstring)
        body_length         : float
        spine_y_center      : int — used for the spine ROI band overlay
        spine_dy            : int — half-height of the spine band
        mean_profile        : 1-D mean intensity profile (smoothed)
        kymograph           : 2-D array [n_strips × W] of the per-strip profiles
        y_spine_original    : 1-D centerline curve y(x) on the ORIGINAL image
        straightened_image  : 2-D float array — the warped image (or the
                              original if straighten=False)
        n_strips            : echoed back
    """
    if image.ndim != 2:
        raise ValueError(f'analyze_image expects a 2-D array, got shape {image.shape}')
    h, w = image.shape

    # ---- fit spine, then straighten so subsequent analysis works on a
    # horizontal fish ----
    y_spine_original, _ = find_spine_centerline(image)
    if straighten:
        work_image, _shifts = straighten_image(image, y_spine_original)
    else:
        work_image = image

    # ---- spine ROI on the (straightened) image ----
    y_center, dy = find_spine_roi(work_image)
    y_lo, y_hi = max(0, y_center - dy), min(h, y_center + dy)
    n_strips = max(2, n_strips)
    y_grid = np.linspace(y_lo, y_hi - 1, n_strips).astype(int)
    half = (n_strips + 1) // 2          # upper-half strip count

    # ---- per-strip peaks ----
    # Detrending: subtract a long-window blur from each strip profile so
    # the slowly-varying baseline (the spine itself is bright everywhere,
    # plus the head is generally brighter than the tail) is removed before
    # peak detection. The somite oscillation sits on top of the baseline
    # with low *relative* prominence — once we subtract the baseline, the
    # oscillation peaks become high-prominence and easy to find. The
    # original (non-detrended) profile is still saved for plotting + kymo.
    strip_peaks: List[Tuple[int, int, float]] = []   # (strip_idx, x, intensity)
    profiles: List[np.ndarray] = []
    detrended_profiles: List[np.ndarray] = []
    for i, y in enumerate(y_grid):
        prof = _strip_profile(work_image, y, strip_thickness, smoothing_sigma)
        profiles.append(prof)
        if detrend_sigma > 0:
            baseline = gaussian_filter1d(prof, sigma=detrend_sigma)
            prof_for_peaks = prof - baseline
        else:
            prof_for_peaks = prof
        detrended_profiles.append(prof_for_peaks)
        for px in _peaks_for_strip(prof_for_peaks, peak_prominence, peak_distance):
            strip_peaks.append((i, int(px), float(prof[px])))

    mean_profile = np.mean(profiles, axis=0) if profiles else np.zeros(w, dtype=np.float32)
    detrended_mean_profile = (np.mean(detrended_profiles, axis=0)
                              if detrended_profiles else np.zeros(w, dtype=np.float32))
    kymograph = np.stack(profiles, axis=0) if profiles else np.zeros((1, w), dtype=np.float32)

    # ---- cluster strip-peaks into somites ----
    clusters = _cluster_peaks(strip_peaks, cluster_window)
    raw_somites: List[Dict] = []
    for cluster in clusters:
        strip_indices = [t[0] for t in cluster]
        xs = [t[1] for t in cluster]
        intensities = [t[2] for t in cluster]
        n_seen = len(set(strip_indices))   # distinct strips that detected this somite
        conf = n_seen / n_strips
        if conf < min_confidence:
            continue
        upper_seen = sum(1 for s in set(strip_indices) if s < half)
        lower_seen = sum(1 for s in set(strip_indices) if s >= half)
        upper_conf = upper_seen / half
        lower_conf = lower_seen / (n_strips - half)
        raw_somites.append({
            'centroid_x': float(np.median(xs)),
            'confidence': float(conf),
            'upper_confidence': float(upper_conf),
            'lower_confidence': float(lower_conf),
            'intensity': float(np.mean(intensities)),
        })

    # ---- order by AP position; assign indices, bbox, severity ----
    raw_somites.sort(key=lambda s: s['centroid_x'])
    if not raw_somites:
        return dict(somites=[], body_length=0.0,
                    spine_y_center=y_center, spine_dy=dy,
                    mean_profile=mean_profile,
                    detrended_mean_profile=detrended_mean_profile,
                    kymograph=kymograph,
                    y_spine_original=y_spine_original,
                    straightened_image=work_image,
                    n_strips=n_strips)

    x_first = raw_somites[0]['centroid_x']
    x_last  = raw_somites[-1]['centroid_x']
    body_length = float(x_last - x_first)
    median_intensity = float(np.median([s['intensity'] for s in raw_somites]))

    somites: List[Dict] = []
    n = len(raw_somites)
    for i, s in enumerate(raw_somites):
        # Per-somite bbox: midway to neighbours (or half the local spacing
        # at the boundaries). y-extent is the spine ROI.
        if i == 0:
            x_left = s['centroid_x'] - (raw_somites[1]['centroid_x'] - s['centroid_x']) / 2 if n > 1 else s['centroid_x'] - 10
        else:
            x_left = (raw_somites[i-1]['centroid_x'] + s['centroid_x']) / 2
        if i == n - 1:
            x_right = s['centroid_x'] + (s['centroid_x'] - raw_somites[i-1]['centroid_x']) / 2 if n > 1 else s['centroid_x'] + 10
        else:
            x_right = (s['centroid_x'] + raw_somites[i+1]['centroid_x']) / 2

        severity, reason = _classify_severity(
            s['confidence'], s['upper_confidence'], s['lower_confidence'],
            s['intensity'], median_intensity)

        somites.append({
            'index': i,
            'centroid_x': s['centroid_x'],
            'centroid_y': float(y_center),
            'bbox': [int(max(0, x_left)), int(y_lo),
                     int(min(w, x_right)), int(y_hi)],
            'ap_position': float((s['centroid_x'] - x_first) /
                                  max(body_length, 1.0)),
            'confidence': s['confidence'],
            'upper_confidence': s['upper_confidence'],
            'lower_confidence': s['lower_confidence'],
            'intensity': s['intensity'],
            'severity': severity,
            'severity_reason': reason,
        })

    return dict(
        somites=somites,
        body_length=body_length,
        spine_y_center=y_center,
        spine_dy=dy,
        mean_profile=mean_profile,
        detrended_mean_profile=detrended_mean_profile,
        kymograph=kymograph,
        y_spine_original=y_spine_original,
        straightened_image=work_image,
        n_strips=n_strips,
    )
