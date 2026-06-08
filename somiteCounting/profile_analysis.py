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
    peak_prominence=0.025,  # in normalised intensity units (0..1) — applied
                            # to the DETRENDED profile, so this is the
                            # somite-oscillation amplitude, not the raw
                            # signal amplitude. Tuned on data/profile_test/
                            # — gives MAE 1.5 on a mix of healthy + defective.
    peak_distance=35,       # min px between peaks. Zebrafish somites are
                            # typically 40-60 px apart in our images; the
                            # parameter sweep on data/profile_test/ showed
                            # this halves the count MAE vs the old default
                            # of 15 (which was eating noise-driven extras).
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


def detect_somites_from_strips(detrended_stack: np.ndarray,
                                mean_profile: np.ndarray,
                                peak_prominence: float = DEFAULTS['peak_prominence'],
                                peak_distance: int = DEFAULTS['peak_distance'],
                                sampling_window: int = DEFAULTS['cluster_window'],
                                evidence_threshold: Optional[float] = None,
                                ) -> List[Dict]:
    """Find somite positions on the mean detrended profile, then score
    per-strip evidence at each position.

    `detrended_stack` is (n_strips, W) — the per-strip high-passed profiles.
    The mean of this stack is fed to `find_peaks`. For each detected peak
    we look at the max value of every strip's profile within
    ±sampling_window of the peak — if that local max exceeds
    `evidence_threshold`, the strip "saw" the somite.

    Returns a list of dicts {centroid_x, confidence, upper_confidence,
    lower_confidence, intensity} — same shape the downstream pipeline
    already consumes; bbox + severity are applied afterwards.

    Splitting detection from per-strip scoring decouples two things that
    used to be tangled: WHERE the somites are (a robust 1-D problem on
    the mean) vs. IS EACH ONE HEALTHY (a per-strip evidence question
    around a now-known x).
    """
    if detrended_stack.size == 0:
        return []
    n_strips, w = detrended_stack.shape

    # Detect on the mean detrended profile
    mean_detrended = detrended_stack.mean(axis=0)
    peaks, _ = find_peaks(mean_detrended,
                           prominence=peak_prominence,
                           distance=peak_distance)
    if len(peaks) == 0:
        return []

    # Adaptive evidence threshold: a peak is "seen" by a strip if the
    # strip's detrended max in the local window exceeds either the
    # caller-supplied threshold OR a fraction of the global prominence
    # (whichever is larger).
    if evidence_threshold is None:
        evidence_threshold = max(peak_prominence * 0.35, 0.003)

    half = (n_strips + 1) // 2          # upper-half strip count

    raw_somites: List[Dict] = []
    for x_peak in peaks:
        x_lo = max(0, int(x_peak) - sampling_window)
        x_hi = min(w, int(x_peak) + sampling_window + 1)
        if x_hi <= x_lo:
            continue
        # Per-strip evidence: each strip's max value in the local window
        per_strip = detrended_stack[:, x_lo:x_hi].max(axis=1)
        seen = per_strip > evidence_threshold
        n_seen = int(seen.sum())
        confidence = n_seen / n_strips
        upper_seen = int(seen[:half].sum())
        lower_seen = int(seen[half:].sum())
        upper_conf = upper_seen / max(half, 1)
        lower_conf = lower_seen / max(n_strips - half, 1)
        intensity = float(mean_profile[int(x_peak)]) if int(x_peak) < len(mean_profile) else 0.0
        raw_somites.append({
            'centroid_x': float(x_peak),
            'confidence': float(confidence),
            'upper_confidence': float(upper_conf),
            'lower_confidence': float(lower_conf),
            'intensity': intensity,
        })
    return raw_somites


def _classify_severity(conf: float, upper: float, lower: float,
                       intensity: float, local_median_intensity: float
                       ) -> Tuple[int, str]:
    """Map (confidence, asymmetry, intensity) → severity 0-3 + reason string.

    `local_median_intensity` is the median of the somite's immediate
    neighbours (not the whole-fish median). Comparing to neighbours is
    important because YFP fluorescence falls off naturally from head to
    tail — every healthy fish has a dim tail-end, and comparing those
    tail somites to the bright head somites used to flag perfectly
    healthy fish as defective.
    """
    asym = abs(upper - lower)
    # Confidence threshold tightened: a true defect tends to show
    # near-zero evidence in many strips, so we only flag confidence
    # below ~40% as "weak detection".
    if conf < 0.40:
        return 3, f'weak detection (conf={conf:.2f})'
    # Asymmetry threshold raised — the previous 0.35 was triggered by
    # ordinary strip-to-strip noise on healthy fish.
    if asym > 0.55:
        return 2, (f'asymmetric: upper={upper:.2f} vs lower={lower:.2f}')
    # Dimness compared to local neighbours, not the global fish.
    if (local_median_intensity > 0
            and intensity < 0.45 * local_median_intensity):
        return 1, (f'dim ({intensity:.2f} vs local {local_median_intensity:.2f})')
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

    # ---- per-strip profiles ----
    # Detrending: subtract a long-window blur from each strip profile so
    # the slowly-varying baseline (the spine itself is bright everywhere,
    # plus the head is generally brighter than the tail) is removed before
    # peak detection. The somite oscillation sits on top of the baseline
    # with low *relative* prominence — once we subtract the baseline, the
    # oscillation peaks become high-prominence and easy to find. The
    # original (non-detrended) profile is still saved for plotting + kymo.
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

    mean_profile = np.mean(profiles, axis=0) if profiles else np.zeros(w, dtype=np.float32)
    detrended_mean_profile = (np.mean(detrended_profiles, axis=0)
                              if detrended_profiles else np.zeros(w, dtype=np.float32))
    kymograph = np.stack(profiles, axis=0) if profiles else np.zeros((1, w), dtype=np.float32)
    detrended_stack = (np.stack(detrended_profiles, axis=0)
                       if detrended_profiles else np.zeros((1, w), dtype=np.float32))

    # ---- find peaks on the MEAN detrended profile ----
    # Chevron arms tilt across y, so the same somite shows up at slightly
    # different x in different strips. Averaging across strips collapses
    # the tilt into one well-defined peak per somite — the mean profile is
    # the cleanest signal we have. Per-strip clustering (the old approach)
    # split each tilted chevron into multiple low-confidence clusters and
    # then rejected them. This approach finds positions on the mean, then
    # *samples* per-strip evidence at each known position — which is much
    # easier than peak-finding in the per-strip signal.
    raw_somites = detect_somites_from_strips(
        detrended_stack, mean_profile,
        peak_prominence=peak_prominence,
        peak_distance=peak_distance,
        sampling_window=cluster_window,
    )
    if not raw_somites:
        return dict(somites=[], body_length=0.0,
                    spine_y_center=y_center, spine_dy=dy,
                    mean_profile=mean_profile,
                    detrended_mean_profile=detrended_mean_profile,
                    detrended_stack=detrended_stack,
                    kymograph=kymograph,
                    y_spine_original=y_spine_original,
                    straightened_image=work_image,
                    n_strips=n_strips)

    x_first = raw_somites[0]['centroid_x']
    x_last  = raw_somites[-1]['centroid_x']
    body_length = float(x_last - x_first)
    # NOTE: per-somite severity is best-effort — intensity-based heuristics
    # cannot reliably distinguish a uniformly-healthy fish from a
    # uniformly-defective one, because both have flat "compared to my
    # neighbours" patterns. For trustworthy defect calls we need a
    # per-somite classifier trained on the bbox tiles we already save.

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

        # Compare intensity to the LOCAL neighbours (a window of ±3
        # somites centred on this one), not the global fish median.
        # Without this, every healthy fish gets its dim-tail somites
        # flagged as defective.
        WIN = 3
        lo_i = max(0, i - WIN)
        hi_i = min(n, i + WIN + 1)
        neighbours = [raw_somites[k]['intensity']
                       for k in range(lo_i, hi_i) if k != i]
        local_median = (float(np.median(neighbours)) if neighbours
                         else float(s['intensity']))

        severity, reason = _classify_severity(
            s['confidence'], s['upper_confidence'], s['lower_confidence'],
            s['intensity'], local_median)

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
        detrended_stack=detrended_stack,
        kymograph=kymograph,
        y_spine_original=y_spine_original,
        straightened_image=work_image,
        n_strips=n_strips,
    )
