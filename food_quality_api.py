"""
food_quality_api.py
Run with:  python food_quality_api.py
"""

import cv2
import numpy as np
import traceback
from flask import Flask, request, jsonify
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
import sklearn

app = Flask(__name__)


# ---------- Utility helpers -------------------------------------------------
def np2py(x):
    """Convert any NumPy scalar to native Python scalar; pass other types unchanged."""
    return x.item() if isinstance(x, np.generic) else x


def safe_kmeans(pixels, n_clusters=3):
    """
    scikit-learn ≥1.2 accepts n_init='auto'; older versions require an int.
    This helper picks the right value automatically.
    """
    minor_ver = int(sklearn.__version__.split(".")[1])
    n_init_val = "auto" if minor_ver >= 2 else 10
    km = KMeans(n_clusters=n_clusters, n_init=n_init_val).fit(pixels)
    return km.cluster_centers_


# ---------- Core analysis ---------------------------------------------------
def analyze_image(image: np.ndarray) -> dict:
    img = cv2.resize(image, (256, 256))
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Texture (GLCM)
    glcm = graycomatrix(gray, [1], [0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]

    # Histogram entropy
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_entropy = entropy(hist / (hist.sum() + 1e-7))

    # Sharpness
    texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # HSV stats
    avg_saturation = np.mean(hsv[:, :, 1])
    avg_hue = np.mean(hsv[:, :, 0]) * 2
    avg_brightness = np.mean(hsv[:, :, 2])

    # Dominant colours
    centers = safe_kmeans(img.reshape(-1, 3)).astype(int).tolist()

    # Dark-spot percent
    _, dark_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    dark_pct = np.sum(dark_mask == 255) / (h * w) * 100.0

    # Shape metrics
    shape_irregularity = circularity = 0.0
    try:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True) or 1
            circularity = 4 * np.pi * area / (peri ** 2)
            hull_area = cv2.contourArea(cv2.convexHull(c)) or 1
            shape_irregularity = (1 - area / hull_area) * 100
    except Exception:
        pass

    # ----- Scoring rules (same logic as before) -----------------------------
    issues, analysis = 0, []

    if texture_score < 200:
        analysis.append("Low sharpness.")
        issues += 1
    if contrast < 5:
        analysis.append("Low texture contrast.")
        issues += 1
    if avg_saturation < 50:
        analysis.append("Dull colour.")
        issues += 1
    if avg_brightness < 50:
        analysis.append("Dark image (possible spoilage).")
        issues += 1
    if dark_pct > 10:
        analysis.append("Dark spots detected.")
        issues += 2
    if circularity < 0.6:
        analysis.append("Irregular shape.")
        issues += 1
    if hist_entropy < 5:
        analysis.append("Low entropy (uniform surface).")
        issues += 1

    if issues >= 5:
        score, verdict = 1, "❌ Not edible — critical defects."
    elif issues >= 3:
        score, verdict = 2, "⚠️ Poor quality."
    elif issues == 2:
        score, verdict = 3, "⚠️ Fair — some concerns."
    elif issues == 1:
        score, verdict = 4, "👌 Good — minor issue."
    else:
        score, verdict = 5, "✅ Excellent."

    return {
        "score": score,
        "recommendation": verdict,
        "metrics": {
            "avg_hue":              np2py(round(avg_hue, 2)),
            "avg_saturation":       np2py(round(avg_saturation, 2)),
            "avg_brightness":       np2py(round(avg_brightness, 2)),
            "texture_score":        np2py(round(texture_score, 2)),
            "contrast":             np2py(round(contrast, 2)),
            "homogeneity":          np2py(round(homogeneity, 2)),
            "entropy":              np2py(round(hist_entropy, 2)),
            "dark_damage_percent":  np2py(round(dark_pct, 2)),
            "shape_irregularity":   np2py(round(shape_irregularity, 2)),
            "circularity":          np2py(round(circularity, 2)),
            "dominant_colors":      centers,
        },
        "analysis": analysis,
    }


# ---------- Flask routes ----------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze_endpoint():
    try:
        if "image" not in request.files:
            return jsonify({"error": "no_image", "msg": "No image uploaded."}), 400

        file_bytes = np.frombuffer(request.files["image"].read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "decode_fail", "msg": "Unsupported or corrupted image."}), 400

        result = analyze_image(img)
        return jsonify(result)

    except Exception as e:
        # log full traceback to console
        traceback.print_exc()
        return jsonify({"error": "internal_error", "msg": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return "🧪 Advanced Food Quality Analyzer API is running.", 200


# ---------- Main ------------------------------------------------------------
if __name__ == "__main__":
    # Listen on all interfaces so Laravel inside WSL / Docker can reach it
    app.run(host="0.0.0.0", port=5000, debug=False)
