#!/usr/bin/env python3
"""
Complete app.py — Flask + OpenCV PCB defect detection (final)
Includes:
- upload or URL image input
- template-based subtraction with alignment (affine -> homography)
- aspect-preserving resize + pad
- automatic rotation correction (0/90/180/270) to match template orientation
- inverse-rotation of outputs so results keep the original uploaded orientation
- stronger preprocessing, morphology and contour filtering
- AJAX /analyze and server-rendered / routes
"""

import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_file, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import requests

# ---------------------------
# Config
# ---------------------------
UPLOAD_FOLDER = "static/uploads"
RESULTS_FOLDER = "static/results"
TEMPLATES_FOLDER = "templates_db"
ALLOWED_EXT = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(TEMPLATES_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "replace-with-secure-key"

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# ---------------------------
# Basic PCB check heuristic
# ---------------------------
def looks_like_pcb_color_edge(img_bgr):
    img = img_bgr.copy()
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 30])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    green_fraction = np.count_nonzero(mask_green) / (h * w)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_fraction = np.count_nonzero(edges) / (h * w)
    return (green_fraction > 0.02) or (edge_fraction > 0.02)

# ---------------------------
# Load a single template (first found)
# ---------------------------
def load_template():
    files = [f for f in os.listdir(TEMPLATES_FOLDER) if allowed_file(f)]
    if not files:
        return None
    path = os.path.join(TEMPLATES_FOLDER, files[0])
    tpl = cv2.imread(path)
    return tpl

# ---------------------------
# Resize & pad helper (preserves aspect ratio)
# ---------------------------
def resize_and_pad(img_bgr, target_w, target_h, pad_color=(0,0,0)):
    h, w = img_bgr.shape[:2]
    if w == 0 or h == 0:
        return None, 1.0
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    if new_w == 0 or new_h == 0:
        return None, scale
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return padded, scale

# ---------------------------
# Rotation helper: pick best of 0/90/180/270 to match template
# Returns rotated_gray, rotated_color, chosen_angle, inverse_angle
# ---------------------------
def best_rotation_match(template_gray, test_color):
    """
    Try rotations (0,90,180,270) of test_color and choose the rotation with the most ORB good matches
    to the template_gray.
    Returns: (rotated_gray, rotated_color, chosen_angle, inverse_angle)
    inverse_angle is the angle to rotate result back to original uploaded orientation.
    """
    orb = cv2.ORB_create(nfeatures=1500)
    kpT, desT = orb.detectAndCompute(template_gray, None)
    if desT is None or len(kpT) < 6:
        gray = cv2.cvtColor(test_color, cv2.COLOR_BGR2GRAY)
        return gray, test_color, 0, 0

    best_angle = 0
    best_score = -1
    best_gray = None
    best_color = None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    for angle in [0, 90, 180, 270]:
        if angle == 0:
            cand_color = test_color.copy()
        else:
            k = (angle // 90) % 4
            cand_color = np.rot90(test_color, k=k)
        cand_gray = cv2.cvtColor(cand_color, cv2.COLOR_BGR2GRAY)
        kpR, desR = orb.detectAndCompute(cand_gray, None)
        if desR is None or len(kpR) < 4:
            continue
        raw = bf.knnMatch(desT, desR, k=2)
        good = 0
        for pair in raw:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good += 1
        if good > best_score:
            best_score = good
            best_angle = angle
            best_gray = cand_gray
            best_color = cand_color

    if best_color is None:
        gray = cv2.cvtColor(test_color, cv2.COLOR_BGR2GRAY)
        return gray, test_color, 0, 0

    inv_angle = (-best_angle) % 360
    return best_gray, best_color, best_angle, inv_angle

# ---------------------------
# Improved align_images: affine first, fallback to homography
# ---------------------------
def align_images(template_gray, test_gray, template_color, test_color,
                 max_features=4000, ratio_thresh=0.65, min_good_matches=12):
    """
    Try affine (estimateAffinePartial2D) first (safer). If not good, try homography.
    Return aligned_gray, aligned_color, transform, valid_mask (or None,... if unreliable).
    """
    try:
        orb = cv2.ORB_create(nfeatures=max_features)
        kp1, des1 = orb.detectAndCompute(template_gray, None)
        kp2, des2 = orb.detectAndCompute(test_gray, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return None, None, None, None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw_matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for pair in raw_matches:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good.append(m)

        if len(good) < min_good_matches:
            return None, None, None, None

        pts_template = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        pts_test = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        h, w = template_gray.shape

        # Try affine (partial) first
        M_affine, inliers_aff = cv2.estimateAffinePartial2D(pts_test.reshape(-1,2), pts_template.reshape(-1,2),
                                                            method=cv2.RANSAC, ransacReprojThreshold=5.0,
                                                            maxIters=2000, confidence=0.99)
        use_affine = False
        if M_affine is not None:
            A = np.vstack([M_affine, [0,0,1]])
            corners = np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]]).T  # 3x4
            warped = (A @ corners).T[:, :2]
            if np.isfinite(warped).all():
                minx, miny = warped.min(axis=0)
                maxx, maxy = warped.max(axis=0)
                warped_area = max(1.0, (maxx-minx)*(maxy-miny))
                template_area = max(1.0, w*h)
                area_ratio = warped_area / template_area
                if 0.3 < area_ratio < 3.5:
                    use_affine = True

        if use_affine:
            med_color = (int(np.median(template_color[:,:,0])),
                         int(np.median(template_color[:,:,1])),
                         int(np.median(template_color[:,:,2])))
            aligned_color = cv2.warpAffine(test_color, M_affine, (w, h),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=med_color)
            aligned_gray = cv2.cvtColor(aligned_color, cv2.COLOR_BGR2GRAY)
            ones = np.ones((test_gray.shape[0], test_gray.shape[1]), dtype=np.uint8) * 255
            mask_aff = cv2.warpAffine(ones, M_affine, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask_aff = (mask_aff > 0).astype(np.uint8) * 255
            return aligned_gray, aligned_color, M_affine, mask_aff

        # Fallback to homography
        H, maskH = cv2.findHomography(pts_test, pts_template, cv2.RANSAC, 5.0, maxIters=2000, confidence=0.99)
        if H is None:
            return None, None, None, None

        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        warped_corners = cv2.perspectiveTransform(corners, H).reshape(-1,2)
        if not np.isfinite(warped_corners).all():
            return None, None, None, None
        min_x, min_y = warped_corners.min(axis=0)
        max_x, max_y = warped_corners.max(axis=0)
        warped_area = max(1.0, (max_x - min_x) * (max_y - min_y))
        template_area = max(1.0, w*h)
        area_ratio = warped_area / template_area
        if area_ratio > 10 or area_ratio < 0.09:
            return None, None, None, None

        med_color = (int(np.median(template_color[:,:,0])),
                     int(np.median(template_color[:,:,1])),
                     int(np.median(template_color[:,:,2])))
        aligned_color = cv2.warpPerspective(test_color, H, (w, h),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=med_color)
        aligned_gray = cv2.cvtColor(aligned_color, cv2.COLOR_BGR2GRAY)
        ones = np.ones_like(test_gray, dtype=np.uint8) * 255
        mask_warp = cv2.warpPerspective(ones, H, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask_warp = (mask_warp > 0).astype(np.uint8) * 255
        return aligned_gray, aligned_color, H, mask_warp

    except Exception as e:
        print("align_images: exception:", e)
        return None, None, None, None

# ---------------------------
# Defect detection (normalize aspect + robust preprocessing + rotation)
# ---------------------------
def detect_defects_by_subtraction(template_color, test_color, min_area=300, max_area=50000, max_scale_change=4.0):
    template_h, template_w = template_color.shape[:2]
    med_color = (int(np.median(template_color[:,:,0])),
                 int(np.median(template_color[:,:,1])),
                 int(np.median(template_color[:,:,2])))

    # 1) Resize & pad to template frame preserving aspect
    test_resized_padded, scale = resize_and_pad(test_color, template_w, template_h, pad_color=med_color)
    if test_resized_padded is None:
        test_resized_padded = cv2.resize(test_color, (template_w, template_h), interpolation=cv2.INTER_LINEAR)

    # 2) Auto-rotation to match template orientation (prevents 90/270 mismatch)
    template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)
    test_gray_rot, test_color_rot, rot_angle, inv_angle = best_rotation_match(template_gray, test_resized_padded)
    print(f"[info] chosen rotation: {rot_angle} degrees, inverse: {inv_angle} degrees")

    # 3) Align (affine/homography)
    aligned_test_gray, aligned_test_color, H, mask = align_images(template_gray, test_gray_rot, template_color, test_color_rot)

    # If alignment failed, fallback to the rotated resized image (already same size as template)
    if aligned_test_gray is None:
        aligned_test_color = test_color_rot
        aligned_test_gray = cv2.cvtColor(aligned_test_color, cv2.COLOR_BGR2GRAY)
        valid_mask = np.ones_like(aligned_test_gray, dtype=np.uint8) * 255
    else:
        ones = np.ones_like(test_gray_rot, dtype=np.uint8) * 255
        if isinstance(H, np.ndarray) and H.shape == (3,3):
            valid_mask = cv2.warpPerspective(ones, H, (template_w, template_h),
                                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            valid_mask = cv2.warpAffine(ones, H, (template_w, template_h),
                                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid_mask = (valid_mask > 0).astype(np.uint8) * 255

    # -----------------------
    # Preprocessing: CLAHE + bilateral to normalize lighting and preserve edges
    # -----------------------
    tpl_gray = template_gray.copy()
    tst_gray = aligned_test_gray.copy()
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        tpl_clahe = clahe.apply(tpl_gray)
        tst_clahe = clahe.apply(tst_gray)
    except Exception:
        tpl_clahe = tpl_gray
        tst_clahe = tst_gray

    tpl_f = cv2.bilateralFilter(tpl_clahe, d=9, sigmaColor=75, sigmaSpace=75)
    tst_f = cv2.bilateralFilter(tst_clahe, d=9, sigmaColor=75, sigmaSpace=75)

    tpl_th = cv2.adaptiveThreshold(tpl_f, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 25, 4)
    tst_th = cv2.adaptiveThreshold(tst_f, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 25, 4)

    if np.mean(tpl_th) < 5 or np.mean(tpl_th) > 250:
        _, tpl_th = cv2.threshold(tpl_f, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(tst_th) < 5 or np.mean(tst_th) > 250:
        _, tst_th = cv2.threshold(tst_f, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # compute diff only inside valid mask
    sub = cv2.absdiff(tpl_th, tst_th)
    sub = cv2.bitwise_and(sub, sub, mask=valid_mask)

    # stronger morphology and border crop to remove edge noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    sub = cv2.medianBlur(sub, 5)
    sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, kernel, iterations=3)
    sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, kernel, iterations=2)

    # erode mask slightly to drop boundary artifacts and crop a small margin
    inner_valid = cv2.erode(valid_mask, np.ones((3,3), np.uint8), iterations=1)
    h_sub, w_sub = sub.shape[:2]
    crop_margin = int(round(min(h_sub, w_sub) * 0.01))  # 1% margin
    if crop_margin > 0:
        crop_mask = np.zeros_like(sub)
        crop_mask[crop_margin:h_sub-crop_margin, crop_margin:w_sub-crop_margin] = 255
        sub = cv2.bitwise_and(sub, sub, mask=crop_mask)
        inner_valid = cv2.bitwise_and(inner_valid, crop_mask)

    sub = cv2.bitwise_and(sub, sub, mask=inner_valid)

    # -----------------------
    # Contour / component filtering
    # -----------------------
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defects = []
    result_img = aligned_test_color.copy()
    mask_out = np.zeros_like(sub)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x,y,w_box,h_box = cv2.boundingRect(c)
        if w_box < 6 or h_box < 6:
            continue
        # solidity check
        comp_mask = np.zeros_like(sub)
        cv2.drawContours(comp_mask, [c], -1, 255, -1)
        contours_c, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_c:
            continue
        cnt = contours_c[0]
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue
        solidity = float(area) / hull_area
        if solidity < 0.35:
            continue
        # accept
        defects.append({"contour": c, "bbox": (x,y,w_box,h_box), "area": area})
        cv2.rectangle(result_img, (x,y), (x+w_box, y+h_box), (0,0,255), 2)
        cv2.drawContours(mask_out, [c], -1, 255, -1)

    # overlay
    overlay = result_img.copy()
    red = np.zeros_like(result_img)
    red[:,:,2] = 255
    overlay = np.where(mask_out[:,:,None] == 255, cv2.addWeighted(result_img, 0.4, red, 0.6, 0), result_img)
    overlay = overlay.astype(np.uint8)

    # -----------------------
    # Inverse-rotate overlay and diff back to original uploaded orientation
    # -----------------------
    if inv_angle != 0:
        k = (inv_angle // 90) % 4
        overlay = np.rot90(overlay, k=k)
        sub = np.rot90(sub, k=k)
        # Note: bounding boxes in `defects` are in aligned coordinates; if you need them in original
        # orientation you'd have to rotate the coordinates — omitted here for simplicity.

    return overlay, len(defects), defects, sub

# ---------------------------
# Save results and build URLs
# ---------------------------
def save_results_and_build_urls(filename_base, overlay_img, diff_img):
    result_name = f"{filename_base}_result.png"
    diff_name   = f"{filename_base}_diff.png"
    result_path = os.path.join(RESULTS_FOLDER, result_name)
    diff_path   = os.path.join(RESULTS_FOLDER, diff_name)
    cv2.imwrite(result_path, overlay_img)
    cv2.imwrite(diff_path, diff_img)
    rel_result_url = url_for('static', filename=f"results/{result_name}")
    rel_diff_url   = url_for('static', filename=f"results/{diff_name}")
    abs_result_url = request.host_url.rstrip('/') + rel_result_url
    abs_diff_url   = request.host_url.rstrip('/') + rel_diff_url
    print("WROTE RESULT FILE:", result_path)
    print("WROTE DIFF FILE:  ", diff_path)
    print("RESULT URL (relative):", rel_result_url)
    print("RESULT URL (absolute):", abs_result_url)
    print("DIFF URL (relative):", rel_diff_url)
    print("DIFF URL (absolute):", abs_diff_url)
    return result_name, diff_name, abs_result_url, abs_diff_url

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        url_val  = request.form.get("image_url", "").strip()
        if not file and not url_val:
            flash("Please upload an image or provide an image URL.")
            return redirect(request.url)

        try:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(save_path)
                img_bgr = cv2.imread(save_path)
            else:
                resp = requests.get(url_val, timeout=8)
                arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                filename = secure_filename(url_val.split("/")[-1]) or "from_url.jpg"
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                cv2.imwrite(save_path, img_bgr)
        except Exception as e:
            print("Error reading/uploading image:", e)
            flash("Could not read the uploaded image or download from URL.")
            return redirect(request.url)

        if img_bgr is None:
            flash("Could not read the uploaded image.")
            return redirect(request.url)

        if not looks_like_pcb_color_edge(img_bgr):
            flash("The uploaded image does not appear to be a PCB. Please upload a PCB image.")
            return redirect(request.url)

        template_color = load_template()
        if template_color is None:
            flash("No template found. Put a non-defective PCB image in templates_db/ named template1.jpg")
            return redirect(request.url)

        overlay, defect_count, defects, diff = detect_defects_by_subtraction(template_color, img_bgr)

        base = os.path.splitext(filename)[0]
        result_name, diff_name, abs_result_url, abs_diff_url = save_results_and_build_urls(base, overlay, diff)

        if defect_count == 0:
            flash("PCB is fine — no defects detected.")
        else:
            flash(f"Defects found: {defect_count}")

        return render_template("index.html",
                               result_image=abs_result_url,
                               diff_image=abs_diff_url,
                               defects=defect_count,
                               result_name=result_name,
                               diff_name=diff_name)

    return render_template("index.html", result_image=None)

@app.route("/analyze", methods=["POST"])
def analyze_api():
    file = request.files.get("image")
    url_val  = request.form.get("image_url", "").strip()
    if not file and not url_val:
        return jsonify(ok=False, message="No file or URL provided."), 400

    try:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)
            img_bgr = cv2.imread(save_path)
        else:
            resp = requests.get(url_val, timeout=8)
            arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            filename = secure_filename(url_val.split("/")[-1]) or "from_url.jpg"
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            cv2.imwrite(save_path, img_bgr)
    except Exception as e:
        return jsonify(ok=False, message=f"Failed to read image: {str(e)}"), 400

    if img_bgr is None:
        return jsonify(ok=False, message="Could not decode image."), 400

    if not looks_like_pcb_color_edge(img_bgr):
        return jsonify(ok=False, message="Uploaded image doesn't appear to be a PCB. Please upload a PCB image."), 400

    template_color = load_template()
    if template_color is None:
        return jsonify(ok=False, message="No template found. Place template1.jpg in templates_db/"), 500

    try:
        overlay, defect_count, defects, diff = detect_defects_by_subtraction(template_color, img_bgr)
    except Exception as e:
        return jsonify(ok=False, message=f"Error during detection: {str(e)}"), 500

    base = os.path.splitext(filename)[0]
    result_name, diff_name, abs_result_url, abs_diff_url = save_results_and_build_urls(base, overlay, diff)

    return jsonify(ok=True, message="Analysis complete",
                   result_image=abs_result_url, diff_image=abs_diff_url,
                   defects=defect_count, result_name=result_name, diff_name=diff_name)

@app.route("/download")
def download():
    name = request.args.get("name")
    if not name:
        flash("File not specified.")
        return redirect(url_for("index"))
    safe_name = secure_filename(name)
    full_path = os.path.join(RESULTS_FOLDER, safe_name)
    if not os.path.exists(full_path):
        flash("File not found.")
        return redirect(url_for("index"))
    return send_file(full_path, as_attachment=True)

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
