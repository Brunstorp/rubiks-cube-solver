"""
Rubik's Cube State Classifier
=============================

Pipeline: 6 photos (one per cube face) -> 54-character state string in the
F U L R D B layout the solver expects.

Output layout (matches the user's solver):

                  |  9 10 11 |
                  | 12 13 14 |   <- U (face 2)
                  | 15 16 17 |
      | 18 19 20 |  0  1  2 | 27 28 29 | 45 46 47 |
      | 21 22 23 |  3  4  5 | 30 31 32 | 48 49 50 |
      | 24 25 26 |  6  7  8 | 33 34 35 | 51 52 53 |
         L (3)      F (1)      R (4)      B (6)
                  | 36 37 38 |
                  | 39 40 41 |   <- D (face 5)
                  | 42 43 44 |

So the returned 54-char string is:
    F[0..8] U[9..17] L[18..26] R[27..35] D[36..44] B[45..53]
and a solved cube prints as
    "FFFFFFFFF" + "UUUUUUUUU" + "LLLLLLLLL" + "RRRRRRRRR" + "DDDDDDDDD" + "BBBBBBBBB"

------------------------------------------------------------------------------
Photo conventions
------------------------------------------------------------------------------
Place 6 files in a folder (default ./pictures), one per face, named by face:

    F.heic   U.heic   L.heic   R.heic   D.heic   B.heic
    (jpg / jpeg / png also work)

For each photo:
    - Hold the cube square to the camera, face roughly fills the frame.
    - Use a low-saturation background (matte black or gray works best).
    - Even, diffuse light. No mixed sources (window + warm bulb is a recipe
      for red-vs-orange confusion).
    - Orientation matters - the rectified face is read row-major (top-left
      first), and the assembled cube must be in URFDLB orientation. The
      simplest convention:
          F, R, L, B  -> hold so the U face is on top of the photo
          U           -> shoot from above with the F edge at the photo's bottom
          D           -> shoot from below with the F edge at the photo's top
      If you took a photo rotated, edit ROTATION below in 90-degree steps.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    pip install opencv-python pillow pillow-heif numpy
    python classify_cube.py ./pictures

or import:
    from classify_cube import classify_cube_from_folder
    state = classify_cube_from_folder("./pictures")

To eyeball that face detection is working before classifying, call
    debug_visualize("./pictures", "F")
which pops up the original with the detected quadrilateral drawn on it,
plus the rectified face with the 3x3 sampling grid.
"""

import sys
from pathlib import Path

import numpy as np
import cv2

# Pillow doesn't read HEIC natively; pillow-heif registers a handler so
# Image.open(...) works on iPhone .heic files. JPG/PNG go through the
# normal Pillow path.
from PIL import Image
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("Warning: pillow-heif not installed; HEIC files won't load.")
    print("    pip install pillow-heif")


# =============================================================================
# Configuration
# =============================================================================

# The 6 face labels, in the exact order the output string expects them.
# The output is the concatenation of these 6 nine-char blocks.
FACE_ORDER = ['F', 'U', 'L', 'R', 'D', 'B']

# Filename extensions to try when looking for each face's photo.
EXTENSIONS = ['.heic', '.HEIC', '.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']

# Side length of the canonical rectified face image (pixels). 300 means each
# of the 9 cells is a 100x100 region we can sample comfortably from.
RECTIFIED_SIZE = 300

# Side length of the patch we sample at the center of each cell. A 40x40
# patch in a 100x100 cell stays well clear of sticker edges and grout lines.
PATCH_SIZE = 40

# Saturation cutoff for white. Real white stickers come back at S < 40 even
# under colored light; colored stickers are typically S > 80. Used by the
# legacy HSV classify_patch; the new clustering pipeline doesn't need it.
WHITE_SAT_THRESHOLD = 50

# If you photographed any face rotated, set its entry here in 90-degree CCW
# steps (1 = rotate 90 CCW, 2 = 180, 3 = 90 CW). Most users won't touch this.
ROTATION = {'F': 0, 'U': 0, 'L': 0, 'R': 0, 'D': 0, 'B': 0}


# =============================================================================
# Image loading
# =============================================================================

def load_image(path: Path) -> np.ndarray:
    """
    Load any supported image format into a BGR numpy array.

    OpenCV doesn't read HEIC, so we go through Pillow (which reads everything
    after pillow-heif registers itself) and then convert RGB -> BGR for cv2.
    """
    pil = Image.open(path).convert('RGB')
    rgb = np.array(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def find_face_image(folder: Path, face: str) -> Path:
    """Find the file in `folder` named `face` with any supported extension."""
    for ext in EXTENSIONS:
        p = folder / f"{face}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No image for face {face!r} in {folder}. "
        f"Looked for {face}{{{', '.join(EXTENSIONS)}}}."
    )


# =============================================================================
# Step 1: find the 4 corners of the cube face in the photo
# =============================================================================

def find_cube_quadrilateral(bgr: np.ndarray) -> np.ndarray:
    """
    Return a (4, 2) array of corners in [TL, TR, BR, BL] order.

    Strategy:
        a) Threshold on saturation in HSV. The cube's stickers are colorful;
           a neutral background isn't. White stickers fail this test (low
           saturation) but they're high-value, so we OR that branch in too.
        b) Morphological close with a kernel sized to the image, so the
           sticker grout lines disappear and the cube reads as one big blob
           rather than 9 disconnected squares.
        c) Take the largest external contour - that's the cube outline.
        d) Approximate it as a polygon. With a clean shot we get exactly 4
           vertices. If we don't, fall back to the min-area rotated rect.
        e) Order the 4 corners as TL, TR, BR, BL using the sum/diff trick.
    """
    # --- (a) saturation+value mask ---------------------------------------
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    # Either colorful (high S) or bright-white (low S, high V). The high-V
    # branch is what catches white stickers without lighting up the whole
    # background if the background is dark.
    mask = ((s > 60) | ((s < 40) & (v > 200))).astype(np.uint8) * 255

    # --- (b) close gaps between stickers ---------------------------------
    # Kernel size scales with image dimension so this works for both 1080p
    # webcam frames and 4032x3024 iPhone photos without tuning.
    img_dim = min(bgr.shape[:2])
    k = max(15, img_dim // 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --- (c) largest contour ---------------------------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError(
            "No cube-like region found. Check the background contrast and "
            "lighting."
        )
    cube = max(contours, key=cv2.contourArea)

    # --- (d) polygon approximation ---------------------------------------
    # epsilon is the max permitted distance between the contour and its
    # approximation; 2% of the perimeter is a well-behaved default for
    # quadrilateral-shaped contours.
    perim = cv2.arcLength(cube, closed=True)
    approx = cv2.approxPolyDP(cube, 0.02 * perim, closed=True)

    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
    else:
        # Fall back: minimum-area rotated rectangle. Less precise (it doesn't
        # care about the actual contour curvature) but always gives 4 points.
        rect = cv2.minAreaRect(cube)
        corners = cv2.boxPoints(rect).astype(np.float32)

    # --- (e) order corners as TL, TR, BR, BL -----------------------------
    # In image coords (y grows downward):
    #   TL has the smallest x + y
    #   BR has the largest  x + y
    #   TR has the largest  x - y
    #   BL has the smallest x - y
    sum_xy = corners.sum(axis=1)
    diff_xy = corners[:, 0] - corners[:, 1]
    tl = corners[np.argmin(sum_xy)]
    br = corners[np.argmax(sum_xy)]
    tr = corners[np.argmax(diff_xy)]
    bl = corners[np.argmin(diff_xy)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


# =============================================================================
# Step 2: warp the cube face into a canonical top-down square
# =============================================================================

def rectify_face(bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Apply a perspective transform that maps `corners` to the corners of a
    RECTIFIED_SIZE x RECTIFIED_SIZE square.

    cv2.getPerspectiveTransform builds the 3x3 homography H satisfying
    H @ [x, y, 1]^T = lambda * [x', y', 1]^T  for each corner pair, and
    cv2.warpPerspective resamples the source into the destination square.
    """
    dst = np.array([
        [0, 0],
        [RECTIFIED_SIZE - 1, 0],
        [RECTIFIED_SIZE - 1, RECTIFIED_SIZE - 1],
        [0, RECTIFIED_SIZE - 1],
    ], dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(bgr, H, (RECTIFIED_SIZE, RECTIFIED_SIZE))


# =============================================================================
# Step 3: sample one (H, S, V) per cell
# =============================================================================

def sample_cells(rectified_bgr: np.ndarray) -> list[tuple[float, float, float]]:
    """
    Return 9 (L, a, b) tuples in row-major order:
        index 0 = top-left, 1 = top-mid, 2 = top-right,
        index 3 = mid-left, 4 = center,  5 = mid-right,
        index 6 = bot-left, 7 = bot-mid, 8 = bot-right.

    OpenCV LAB: L in [0, 255], a/b in [0, 255] with 128 = neutral grey.
    Median (not mean) is robust to glare specks and sticker edges that
    happen to fall inside the patch.

    LAB is preferred over HSV here because (a, b) is roughly perceptually
    uniform — the red/orange separation that's tiny in HSV-hue space is
    ~25 units in LAB, well above the noise floor.
    """
    cell = RECTIFIED_SIZE // 3      # 100 with the default
    half = PATCH_SIZE // 2

    lab = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2LAB)
    out = []
    for row in range(3):
        for col in range(3):
            cx = col * cell + cell // 2
            cy = row * cell + cell // 2
            patch = lab[cy - half:cy + half, cx - half:cx + half]
            L = float(np.median(patch[:, :, 0]))
            a = float(np.median(patch[:, :, 1]))
            b = float(np.median(patch[:, :, 2]))
            out.append((L, a, b))
    return out


# =============================================================================
# Step 4: classify each patch against per-session reference centroids
# =============================================================================

def hue_distance(h1: float, h2: float) -> float:
    """
    Circular distance on OpenCV's [0, 180] hue scale. Going from red (~0) to
    red-just-past-purple (~179) should be 1, not 179.
    """
    d = abs(h1 - h2)
    return min(d, 180.0 - d)


# Below this LAB-distance gap (best-cluster vs second-best-cluster), we
# flag a sticker as ambiguous. In LAB (a, b) Euclidean distance the colour
# clusters are typically 25-80 apart; a gap below 6 means the sticker sits
# between two cluster centres.
UNCERTAIN_GAP_THRESHOLD = 6.0


def _kmeans(points: np.ndarray, k: int, n_iter: int = 60, seed: int = 0):
    """
    Tiny k-means++ for low-d clustering. Returns (labels, dists) where
    dists is shape (n_points, k) — each entry is the distance from that
    point to that cluster centre.

    k-means++ initialisation (each next centre picked with probability ∝
    distance² from any chosen) gives much better seedings than random init
    for well-separated clusters like cube colours.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < k:
        raise ValueError(f"need at least {k} points to cluster, got {n}")
    rng = np.random.default_rng(seed)

    # k-means++ init.
    centres = [points[rng.integers(n)]]
    for _ in range(k - 1):
        d2 = np.min(
            np.sum((points[:, None] - np.asarray(centres)[None, :]) ** 2, axis=2),
            axis=1,
        )
        total = d2.sum()
        if total <= 0:
            idx = int(rng.integers(n))
        else:
            idx = int(rng.choice(n, p=d2 / total))
        centres.append(points[idx])
    centres = np.array(centres)

    # Lloyd iterations.
    for _ in range(n_iter):
        d = np.linalg.norm(points[:, None] - centres[None, :], axis=2)
        labels = np.argmin(d, axis=1)
        new_centres = np.array([
            points[labels == i].mean(axis=0) if (labels == i).any() else centres[i]
            for i in range(k)
        ])
        if np.allclose(new_centres, centres, atol=0.5):
            break
        centres = new_centres

    d = np.linalg.norm(points[:, None] - centres[None, :], axis=2)
    return np.argmin(d, axis=1), d


def classify_patch(patch_lab, references: dict) -> str:
    """[backward compat] Nearest reference in LAB (a, b)."""
    _L, a, b = patch_lab
    return min(
        references.keys(),
        key=lambda k: (references[k][1] - a) ** 2 + (references[k][2] - b) ** 2,
    )


# =============================================================================
# Per-face glue
# =============================================================================

def process_face(folder: Path, face: str):
    """
    Load one face's photo, rectify it, sample 9 patches.

    Returns (patches, rectified_bgr) so callers can debug-display the
    rectified image if they want.
    """
    path = find_face_image(folder, face)
    bgr = load_image(path)

    # Per-face rotation (in case someone shot a face sideways).
    k = ROTATION.get(face, 0) % 4
    if k:
        bgr = np.ascontiguousarray(np.rot90(bgr, k=k))

    corners = find_cube_quadrilateral(bgr)
    rectified = rectify_face(bgr, corners)
    patches = sample_cells(rectified)
    return patches, rectified


# =============================================================================
# Top-level pipeline
# =============================================================================

def classify_cube_from_folder(folder) -> str:
    """
    End-to-end: read 6 photos from disk, return a 54-character cube state.
    Thin wrapper around classify_cube_from_images.
    """
    folder = Path(folder)
    images = {face: load_image(find_face_image(folder, face)) for face in FACE_ORDER}
    return classify_cube_from_images(images)


def _build_classification(images: dict):
    """
    Cluster all 54 stickers in LAB (a, b) space, then label each cluster
    via that face's centre patch. Returns (state, uncertain_idx, warning).

    Why this is more robust than per-face nearest-reference:
      * Lighting cast that's common across photos shifts every sticker by
        the same amount, so the relative cluster structure survives. We
        don't have to guess the illuminant.
      * 6 well-separated colour clusters appear naturally; k-means++ init
        latches onto them reliably.
      * Each face's centre is the only thing we trust — and it's used for
        labelling, not for measuring distance.

    Best-effort behaviour as before: ≥3 failed photos → raise; otherwise
    fill missing faces with their own letter and flag everything.
    """
    all_patches = {}
    failed_faces = []
    for face in FACE_ORDER:
        if face not in images:
            raise ValueError(f"missing image for face {face!r}")
        bgr = images[face]
        k = ROTATION.get(face, 0) % 4
        if k:
            bgr = np.ascontiguousarray(np.rot90(bgr, k=k))
        try:
            corners = find_cube_quadrilateral(bgr)
            rectified = rectify_face(bgr, corners)
            all_patches[face] = sample_cells(rectified)
        except Exception:
            all_patches[face] = None
            failed_faces.append(face)

    successful = [f for f in FACE_ORDER if all_patches[f] is not None]
    if len(successful) < 4:
        raise ValueError(
            f"could not detect a cube in {len(failed_faces)} of 6 photos "
            f"({', '.join(failed_faces)}). Retake with better lighting and a "
            f"plain background."
        )

    # Flatten all successful stickers into a single (a, b) feature matrix.
    # We drop L (lightness) — it varies way too much with shadows / glare
    # to be useful, and the colour separation lives entirely in (a, b).
    flat_features = []
    flat_idx = []  # global sticker index 0..53
    for face_pos, face in enumerate(FACE_ORDER):
        if all_patches[face] is None:
            continue
        for cell_idx, (_L, a, b) in enumerate(all_patches[face]):
            flat_features.append((a, b))
            flat_idx.append(face_pos * 9 + cell_idx)

    # Always cluster into 6 groups, even if a face's photo is missing — the
    # remaining 45 stickers still cover all 6 colours (each colour appears
    # 9× somewhere on a real cube), and k=6 gives k-means++ enough room to
    # find them all.
    labels, dists = _kmeans(np.asarray(flat_features), k=6)

    # Map clusters → faces by where each face's CENTRE landed.
    cluster_to_face = {}
    for j, sticker_idx in enumerate(flat_idx):
        face_pos, cell_idx = divmod(sticker_idx, 9)
        if cell_idx == 4:
            cluster_to_face[int(labels[j])] = FACE_ORDER[face_pos]

    # Fill in any missing face from the leftover unmapped cluster (single
    # missing face only — with two we can't disambiguate).
    unmapped = set(range(6)) - set(cluster_to_face.keys())
    if len(unmapped) == 1 and len(failed_faces) == 1:
        cluster_to_face[unmapped.pop()] = failed_faces[0]

    # Build per-face label arrays.
    labels_by_face = {f: ['?'] * 9 for f in FACE_ORDER}
    uncertain_idx = []

    # Failed faces: fill with the face's own letter, flag every non-centre.
    for face_pos, face in enumerate(FACE_ORDER):
        if all_patches[face] is None:
            labels_by_face[face] = [face] * 9
            for c_idx in range(9):
                if c_idx != 4:
                    uncertain_idx.append(face_pos * 9 + c_idx)

    # Successful faces: cluster → face.
    for j, sticker_idx in enumerate(flat_idx):
        face_pos, cell_idx = divmod(sticker_idx, 9)
        cluster = int(labels[j])
        face_label = cluster_to_face.get(cluster)
        if face_label is None:
            # Cluster unmapped (≥2 failed faces). Fall back to the nearest
            # mapped cluster and flag.
            for c in np.argsort(dists[j]):
                if int(c) in cluster_to_face:
                    face_label = cluster_to_face[int(c)]
                    break
            face_label = face_label or FACE_ORDER[face_pos]
            uncertain_idx.append(sticker_idx)
        labels_by_face[FACE_ORDER[face_pos]][cell_idx] = face_label

        # Confidence gap: distance to second-nearest cluster minus distance
        # to assigned cluster. Centres are tautologically right.
        if cell_idx != 4:
            sorted_d = np.sort(dists[j])
            if (sorted_d[1] - sorted_d[0]) < UNCERTAIN_GAP_THRESHOLD:
                uncertain_idx.append(sticker_idx)

    state = "".join("".join(labels_by_face[f]) for f in FACE_ORDER)
    warning = _soft_validate(state, uncertain_idx, failed_faces)
    uncertain_idx = sorted(set(uncertain_idx))
    return state, uncertain_idx, warning


def _soft_validate(state, uncertain_idx, failed_faces):
    """Describe remaining problems and bulk-flag obviously-suspect stickers.
    Mutates uncertain_idx in place. Returns the warning string or None."""
    issues = []
    if failed_faces:
        issues.append(
            f"couldn't detect a cube in the {', '.join(failed_faces)} photo(s); "
            "those faces were filled with the face colour"
        )
    counts = {face: state.count(face) for face in FACE_ORDER}
    bad = [face for face in FACE_ORDER if counts[face] != 9]
    if bad:
        parts = [f"{face}={counts[face]}" for face in bad]
        issues.append(f"colour counts off: {', '.join(parts)} (want 9 each)")
        # Flag every non-centre sticker of any wrong-count colour so the user
        # knows where to look. Some are right and some are wrong; we can't
        # tell which without their input.
        for idx, c in enumerate(state):
            if c in bad and (idx % 9) != 4:
                uncertain_idx.append(idx)
    if not issues:
        return None
    return ("Capture issues: " + "; ".join(issues)
            + ". Repaint the highlighted stickers, then Solve.")


def classify_cube_from_images(images: dict) -> str:
    """End-to-end (strict): 6 in-memory BGR ndarrays -> 54-letter state.
    Raises if the result doesn't pass strict count / centre validation."""
    state, _, warning = _build_classification(images)
    if warning:
        # Match the pre-best-effort behaviour for callers that want strict
        # validation (the CLI uses this).
        raise ValueError(warning)
    return state


def classify_cube_from_images_with_uncertain(images: dict):
    """End-to-end (lenient): returns (state, uncertain_indices, warning).
    state is always a 54-letter string. warning is a user-facing string
    describing problems the user needs to fix manually, or None."""
    return _build_classification(images)


# Reference-separation pre-flight was an HSV-era diagnostic; the new
# clustering pipeline catches confused colours via the per-sticker
# uncertainty gap, so an explicit pre-check isn't useful.


# =============================================================================
# Per-sticker contour detection for live-webcam scanning
# =============================================================================
# Algorithm follows https://github.com/exactful/rubiks-cube-face-detection:
# Canny edges → dilate → findContours → keep 4-sided squarish blobs of the
# right size whose dominant colour is close to one of 6 fixed references →
# require 9 such contours arranged around a centre. Returns the 9 stickers
# in 3×3 row-major order for the current face, or None if the frame is
# unusable.

# Standard Western cube colour scheme in BGR.
_REF_BGR = {
    'F': ( 75, 178,  76),   # green
    'U': (255, 255, 255),   # white
    'L': (  0, 122, 255),   # orange
    'R': (  0,   0, 215),   # red
    'D': (  0, 213, 255),   # yellow
    'B': (214,  77,  26),   # blue
}


def _refs_lab():
    """Cache the LAB conversion of the reference colours (depends on cv2)."""
    if not hasattr(_refs_lab, "_cached"):
        out = {}
        for face, bgr in _REF_BGR.items():
            pixel = np.uint8([[list(bgr)]])
            lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0]
            out[face] = (float(lab[0]), float(lab[1]), float(lab[2]))
        _refs_lab._cached = out
    return _refs_lab._cached


def _dominant_bgr(image: np.ndarray) -> tuple[int, int, int]:
    """OpenCV k-means k=1 on the contour interior, i.e. the BGR colour that
    minimises in-cluster squared distance — essentially the central tendency
    minus glare/edges. Faster + more robust than np.mean for noisy patches."""
    data = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centres = cv2.kmeans(data, 1, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    return tuple(int(x) for x in centres[0])


def _match_sticker_colour(bgr, max_delta=70.0):
    """Closest reference face label by LAB Euclidean. Returns (face, delta)
    or (None, delta) when the closest reference is too far to be plausible —
    that's how non-cube contours (skin, walls, labels) get rejected."""
    pixel = np.uint8([[list(bgr)]])
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0]
    best, best_d = None, float('inf')
    for face, ref in _refs_lab().items():
        d = ((float(lab[0]) - ref[0]) ** 2
             + (float(lab[1]) - ref[1]) ** 2
             + (float(lab[2]) - ref[2]) ** 2) ** 0.5
        if d < best_d:
            best_d, best = d, face
    return (best if best_d < max_delta else None), best_d


def detect_stickers(bgr: np.ndarray):
    """Return 9 sticker dicts {x, y, w, h, color} in 3×3 row-major order, or
    None if the frame doesn't have a recognisable cube face.

    color is one of 'F','U','L','R','D','B' (Western scheme); coordinates
    are in pixels of the input image."""
    if bgr is None or bgr.size == 0:
        return None

    h_img, w_img = bgr.shape[:2]
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    noiseless = cv2.fastNlMeansDenoising(grey, None, 20, 7, 7)
    blurred = cv2.blur(noiseless, (3, 3))
    edges = cv2.Canny(blurred, 30, 60, 3)
    dilated = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
    )
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sticker-size band scales with image so it works for both 640×480 and
    # 1080p frames without retuning. A sticker on a centred cube fills
    # roughly 4 %–40 % of the smaller image dimension.
    short_side = min(h_img, w_img)
    min_w = max(20, int(short_side * 0.04))
    max_w = max(min_w + 1, int(short_side * 0.40))
    min_area = max(400, int((short_side * 0.04) ** 2))

    candidates = []
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.1 * cv2.arcLength(c, True), True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        ratio = float(w) / max(h, 1)
        area = cv2.contourArea(approx)
        if not (0.8 <= ratio <= 1.2 and min_w <= w <= max_w and area >= min_area):
            continue
        # Inset the patch slightly so we sample the sticker interior, not the
        # darker bevel edge.
        ix = max(0, x + int(w * 0.15))
        iy = max(0, y + int(h * 0.15))
        iw = max(1, int(w * 0.70))
        ih = max(1, int(h * 0.70))
        patch = bgr[iy:iy + ih, ix:ix + iw]
        if patch.size == 0:
            continue
        dom = _dominant_bgr(patch)
        face, _ = _match_sticker_colour(dom)
        if face is None:
            continue
        candidates.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "color": face,
            "bgr": [int(dom[0]), int(dom[1]), int(dom[2])],
        })

    if len(candidates) < 9:
        return None

    # Too many? Keep the 9 closest to the median position (median is robust
    # to a handful of background squares that survived the size+colour
    # filter).
    if len(candidates) > 9:
        med_x = float(np.median([c["x"] + c["w"] / 2 for c in candidates]))
        med_y = float(np.median([c["y"] + c["h"] / 2 for c in candidates]))
        candidates.sort(
            key=lambda c: (c["x"] + c["w"] / 2 - med_x) ** 2
                          + (c["y"] + c["h"] / 2 - med_y) ** 2
        )
        candidates = candidates[:9]

    # Sort into 3×3: bucket by y, then x within each row.
    by_y = sorted(candidates, key=lambda c: c["y"])
    rows = []
    for i in range(0, 9, 3):
        rows.extend(sorted(by_y[i:i + 3], key=lambda c: c["x"]))

    # Spatial sanity check: the 4 extreme stickers must be within ~1.7×
    # sticker-width of the middle one. Rejects scenes where some "stickers"
    # are background noise that survived the earlier filters.
    middle = rows[4]
    by_x = sorted(rows, key=lambda c: c["x"])
    gap_w = int(middle["w"] * 1.7)
    gap_h = int(middle["h"] * 1.7)
    if not (middle["x"] - by_x[0]["x"] <= gap_w
            and by_x[-1]["x"] - middle["x"] <= gap_w
            and middle["y"] - by_y[0]["y"] <= gap_h
            and by_y[-1]["y"] - middle["y"] <= gap_h):
        return None

    return rows


# Validation lives in _soft_validate (called by _build_classification); the
# old strict _validate_state was removed when the pipeline switched to
# best-effort. Callers that want strict behaviour should use
# classify_cube_from_images, which raises on warning.


# =============================================================================
# Optional debug visualization
# =============================================================================

def debug_visualize(folder, face: str):
    """
    Pop up two windows for a single face:
      - the original photo with the detected quadrilateral drawn in green
      - the rectified face with the 3x3 sampling grid overlaid

    Useful for diagnosing bad face detection before you start blaming the
    classifier.
    """
    folder = Path(folder)
    path = find_face_image(folder, face)
    bgr = load_image(path)
    k = ROTATION.get(face, 0) % 4
    if k:
        bgr = np.ascontiguousarray(np.rot90(bgr, k=k))

    corners = find_cube_quadrilateral(bgr)

    overlay = bgr.copy()
    pts = corners.astype(int).reshape(-1, 1, 2)
    cv2.polylines(overlay, [pts], True, (0, 255, 0), thickness=8)

    rectified = rectify_face(bgr, corners)
    rect_overlay = rectified.copy()
    cell = RECTIFIED_SIZE // 3
    for i in range(1, 3):
        cv2.line(rect_overlay, (i * cell, 0), (i * cell, RECTIFIED_SIZE),
                 (255, 255, 255), 2)
        cv2.line(rect_overlay, (0, i * cell), (RECTIFIED_SIZE, i * cell),
                 (255, 255, 255), 2)

    # Resize the original for screen-friendly display.
    h_disp = 800
    scale = h_disp / overlay.shape[0]
    disp = cv2.resize(overlay, None, fx=scale, fy=scale)

    cv2.imshow(f"{face} - detected quad (green)", disp)
    cv2.imshow(f"{face} - rectified", rect_overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "./pictures"
    state = classify_cube_from_folder(folder)
    print(state)