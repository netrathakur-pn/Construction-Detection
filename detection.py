"""
detection.py  — v6  (Clean Patch Visualisation)
─────────────────────────────────────────────────────────────────────────────
Change in v6:
  • Result image now shows the satellite image clearly with ONLY the detected
    change regions coloured as clean distinct patches (green=authorised,
    red=unauthorised). The rest of the image is the original satellite photo.
  • Previously the entire image was tinted which made it hard to see.

Full pipeline:
  Load → Align (ORB) → Deep Feature Diff (ResNet50 / OpenCV fallback) →
  Threshold → Morph Clean → Building Merge → Contour Filter →
  SSIM → Colour Check → OSM Overlap → Classify →
  Clean patch render → Save outputs
"""

import cv2
import math
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ── PyTorch (optional) ────────────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    import torchvision.models    as tvmodels
    import torchvision.transforms as tvT
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════
# 1.  Deep Feature Extractor
# ══════════════════════════════════════════════════════════════════════════

class _FeatureExtractor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._loaded = False
            cls._instance = obj
        return cls._instance

    def _load(self):
        if self._loaded:
            return
        try:
            model = tvmodels.resnet50(weights=tvmodels.ResNet50_Weights.DEFAULT)
        except AttributeError:
            model = tvmodels.resnet50(pretrained=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(self.device)
        self.model  = model
        self._cache = {}
        self.model.layer2.register_forward_hook(self._hook("layer2"))
        self.model.layer3.register_forward_hook(self._hook("layer3"))
        self.tf = tvT.Compose([
            tvT.ToTensor(),
            tvT.Normalize(mean=[0.485, 0.456, 0.406],
                          std =[0.229, 0.224, 0.225]),
        ])
        self._loaded = True

    def _hook(self, name):
        def fn(_, __, out):
            self._cache[name] = out.detach()
        return fn

    def _to_tensor(self, bgr):
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            rgb   = cv2.resize(rgb, (int(w*scale), int(h*scale)),
                               interpolation=cv2.INTER_AREA)
        return self.tf(rgb).unsqueeze(0).to(self.device)

    def change_map(self, img1, img2):
        self._load()
        orig_h, orig_w = img1.shape[:2]
        feat = []
        with torch.no_grad():
            for img in [img1, img2]:
                self.model(self._to_tensor(img))
                feat.append({k: v.clone() for k, v in self._cache.items()})
        f1, f2   = feat
        combined = []
        for key in ["layer2", "layer3"]:
            dist = torch.norm(f1[key] - f2[key], p=2, dim=1, keepdim=True)
            up   = F.interpolate(dist, size=(orig_h, orig_w),
                                 mode="bilinear", align_corners=False)
            combined.append(up.squeeze().cpu().numpy())
        out    = 0.5 * combined[0] + 0.5 * combined[1]
        mn, mx = out.min(), out.max()
        if mx - mn > 1e-8:
            out = (out - mn) / (mx - mn)
        return (out * 255).astype(np.uint8)


_extractor = _FeatureExtractor() if TORCH_AVAILABLE else None


# ══════════════════════════════════════════════════════════════════════════
# 2.  Image Alignment
# ══════════════════════════════════════════════════════════════════════════

def align_images(ref, src, max_feat=5000, ratio=0.75):
    h, w     = ref.shape[:2]
    g1, g2   = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in [ref, src])
    orb      = cv2.ORB_create(max_feat)
    kp1, d1  = orb.detectAndCompute(g1, None)
    kp2, d2  = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(kp1) < 10 or len(kp2) < 10:
        return src, False
    bf   = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = [m for m, n in bf.knnMatch(d1, d2, k=2)
            if len((m, n)) == 2 and m.distance < ratio * n.distance]
    if len(good) < 10:
        return src, False
    p1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    p2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, _ = cv2.findHomography(p2, p1, cv2.RANSAC, 5.0)
    if H is None:
        return src, False
    return (cv2.warpPerspective(src, H, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT), True)


# ══════════════════════════════════════════════════════════════════════════
# 3.  OpenCV Fallback Diff
# ══════════════════════════════════════════════════════════════════════════

def _opencv_diff(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    l1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)[:,:,0].astype(np.float32)
    l2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)[:,:,0].astype(np.float32)
    return np.clip(0.6*np.abs(g1-g2) + 0.4*np.abs(l1-l2), 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════
# 4.  Filters
# ══════════════════════════════════════════════════════════════════════════

def _building_shaped(c, area):
    x, y, bw, bh = cv2.boundingRect(c)
    if bw == 0 or bh == 0:
        return False
    return (max(bw,bh)/min(bw,bh) <= 6) and (area/(bw*bh) >= 0.20)


def _construction_colour(patch):
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue = float(np.mean(hsv[:,:,0]))
    sat = float(np.mean(hsv[:,:,1]))
    val = float(np.mean(hsv[:,:,2]))
    if 35 < hue < 85 and sat > 60: return False
    if val < 25:                    return False
    return True


# ══════════════════════════════════════════════════════════════════════════
# 5.  Area Conversion
# ══════════════════════════════════════════════════════════════════════════

def pixels_to_sqm(pixel_area, lat_min, lat_max, lon_min, lon_max,
                  img_w, img_h):
    if any(v is None for v in [lat_min, lat_max, lon_min, lon_max]):
        return None
    if img_w == 0 or img_h == 0:
        return None
    lat_c        = (lat_min + lat_max) / 2.0
    m_per_lat    = 111_320.0
    m_per_lon    = 111_320.0 * math.cos(math.radians(lat_c))
    img_w_m      = abs(lon_max - lon_min) * m_per_lon
    img_h_m      = abs(lat_max - lat_min) * m_per_lat
    return round(pixel_area * (img_w_m * img_h_m) / (img_w * img_h), 1)


# ══════════════════════════════════════════════════════════════════════════
# 6.  Main Detection Function
# ══════════════════════════════════════════════════════════════════════════

def detect_changes(before_path, after_path, result_path,
                   permit_mask=None,
                   lat_min=None, lat_max=None,
                   lon_min=None, lon_max=None):

    has_gps = all(v is not None for v in [lat_min, lat_max, lon_min, lon_max])

    # ── Load ──────────────────────────────────────────────────────────────
    before = cv2.imread(before_path)
    after  = cv2.imread(after_path)
    if before is None or after is None:
        raise ValueError("Cannot read images. Check file paths.")

    h, w  = before.shape[:2]
    after = cv2.resize(after, (w, h))
    if permit_mask is not None:
        permit_mask = cv2.resize(permit_mask, (w, h),
                                 interpolation=cv2.INTER_NEAREST)

    # ── Step 1: Align ─────────────────────────────────────────────────────
    af, align_ok = align_images(before, after)
    align_status = ("✅ ORB alignment successful."
                    if align_ok else
                    "⚠️ Alignment skipped — few matching features.")

    # ── Step 2: Change map ────────────────────────────────────────────────
    if TORCH_AVAILABLE:
        method = "ResNet50 Deep Feature Difference (pre-trained, ImageNet)"
        try:
            cmap = _extractor.change_map(before, af)
        except Exception as e:
            method = f"OpenCV diff (PyTorch error: {e})"
            cmap   = _opencv_diff(before, af)
    else:
        method = "OpenCV multi-channel diff  [install torch for higher accuracy]"
        cmap   = _opencv_diff(before, af)

    # ── Step 3: Threshold ─────────────────────────────────────────────────
    blurred = cv2.GaussianBlur(cmap, (9, 9), 0)
    if TORCH_AVAILABLE:
        _, thresh = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)

    # ── Step 4: Sanity check ──────────────────────────────────────────────
    raw_ratio = float(np.sum(thresh > 0)) / thresh.size * 100
    if raw_ratio > 65.0:
        err = before.copy()
        cv2.putText(err, "IMAGES TOO MISALIGNED", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3, cv2.LINE_AA)
        cv2.putText(err,
                    f"{raw_ratio:.1f}% pixels differ — re-export same GEP view.",
                    (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,200,255), 2, cv2.LINE_AA)
        cv2.putText(err, "Press U in GEP for top-down view first.",
                    (20,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,200,255), 2, cv2.LINE_AA)
        cv2.imwrite(result_path, err)
        return err, {
            "verdict":"ERROR",
            "error_msg":(f"Images too misaligned ({raw_ratio:.1f}% pixels changed). "
                         "Re-export both from the EXACT same GEP view. "
                         "Press U for top-down view first."),
            "total_changes":0, "authorised_count":0, "unauthorised_count":0,
            "authorised_area_px":0, "unauthorised_area_px":0,
            "authorised_area_sqm":None, "unauthorised_area_sqm":None,
            "buildings":[], "change_ratio_percent":round(raw_ratio,2),
            "changed_pixels":int(np.sum(thresh>0)),
            "alignment_status":align_status, "method_used":method,
            "mask_path":result_path, "mask_auth_path":result_path,
            "mask_unauth_path":result_path, "osm_overlay_path":None,
            "change_map_path":None,
        }

    # ── Step 5: Morphological cleaning ────────────────────────────────────
    clean  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  np.ones((9, 9),  np.uint8))
    clean  = cv2.morphologyEx(clean,  cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))

    # ── Step 5b: Building merge ───────────────────────────────────────────
    merge_k = np.ones((25, 25), np.uint8)
    merged  = cv2.dilate(clean, merge_k, iterations=1)
    merged  = cv2.erode(merged, merge_k, iterations=1)

    # ── Step 6: Contour analysis ──────────────────────────────────────────
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    mask_auth   = np.zeros((h, w), dtype=np.uint8)
    mask_unauth = np.zeros((h, w), dtype=np.uint8)

    auth_count      = 0
    unauth_count    = 0
    auth_areas_px   = []
    unauth_areas_px = []
    buildings       = []

    gray_b  = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_a  = cv2.cvtColor(af,     cv2.COLOR_BGR2GRAY)
    MIN_AREA = 1500

    valid_contours = []   # contours that pass all filters

    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        if not _building_shaped(c, area):
            continue

        x, y, cw, ch = cv2.boundingRect(c)
        pb, pa = gray_b[y:y+ch, x:x+cw], gray_a[y:y+ch, x:x+cw]
        if pb.shape != pa.shape or pb.size < 64:
            continue
        try:
            sim = ssim(pb, pa, data_range=255)
        except Exception:
            sim = 1.0
        if sim >= 0.78:
            continue

        if not _construction_colour(af[y:y+ch, x:x+cw]):
            continue

        area_sqm = pixels_to_sqm(area, lat_min, lat_max,
                                  lon_min, lon_max, w, h)

        cmask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cmask, [c], -1, 255, -1)
        cpx = int(np.sum(cmask > 0))

        if permit_mask is not None:
            ov      = int(np.sum(cv2.bitwise_and(cmask, permit_mask) > 0))
            is_auth = (ov / cpx >= 0.30) if cpx > 0 else False
        else:
            is_auth = False

        sqm_label = f" ~{int(area_sqm)}m²" if area_sqm else ""

        if is_auth:
            auth_count += 1
            auth_areas_px.append(area)
            cv2.drawContours(mask_auth, [c], -1, 255, -1)
            status = "AUTHORISED"
        else:
            unauth_count += 1
            unauth_areas_px.append(area)
            cv2.drawContours(mask_unauth, [c], -1, 255, -1)
            status = "UNAUTHORISED"

        valid_contours.append((c, is_auth, sqm_label, x, y, cw, ch))

        buildings.append({
            "id":       idx + 1,
            "status":   status,
            "area_px":  int(area),
            "area_sqm": area_sqm,
            "x": x, "y": y, "w": cw, "h": ch,
        })

    # ── Step 7: Clean patch visualisation ────────────────────────────────
    # Start with the original satellite image — clear background
    result_overlay = af.copy()

    # Build a colour layer — only fill detected regions
    colour_layer = np.zeros_like(af)
    colour_layer[mask_auth   > 0] = (0,  200,  0)   # green
    colour_layer[mask_unauth > 0] = (0,    0, 200)   # red

    # Blend colour ONLY on detected patch pixels — rest stays as satellite image
    patch_pixels = (mask_auth > 0) | (mask_unauth > 0)
    result_overlay[patch_pixels] = cv2.addWeighted(
        colour_layer, 0.65, af, 0.35, 0
    )[patch_pixels]

    # Draw clean outlines + labels on patches
    for (c, is_auth, sqm_label, x, y, cw, ch) in valid_contours:
        if is_auth:
            outline_col = (0, 255, 0)
            label       = f"AUTH{sqm_label}"
        else:
            outline_col = (0, 0, 255)
            label       = f"UNAUTH{sqm_label}"

        cv2.drawContours(result_overlay, [c], -1, outline_col, 2)
        cv2.rectangle(result_overlay, (x, y), (x+cw, y+ch), outline_col, 1)

        # Label background for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        ty = max(y - 4, th + 4)
        cv2.rectangle(result_overlay,
                      (x, ty - th - 3), (x + tw + 4, ty + 2),
                      outline_col, -1)
        cv2.putText(result_overlay, label, (x + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255),
                    1, cv2.LINE_AA)

    # ── Step 8: Verdict banner ────────────────────────────────────────────
    total = auth_count + unauth_count

    total_auth_sqm   = (pixels_to_sqm(sum(auth_areas_px),
                                      lat_min, lat_max, lon_min, lon_max, w, h)
                        if auth_areas_px and has_gps else None)
    total_unauth_sqm = (pixels_to_sqm(sum(unauth_areas_px),
                                      lat_min, lat_max, lon_min, lon_max, w, h)
                        if unauth_areas_px and has_gps else None)

    if total == 0:
        verdict, btxt, bcol = (
            "NO_CHANGE",
            "NO SIGNIFICANT CONSTRUCTION CHANGES DETECTED",
            (55,55,55))
    elif unauth_count > 0 and auth_count == 0:
        verdict  = "UNAUTHORISED"
        area_str = f" — ~{int(total_unauth_sqm)} m²" if total_unauth_sqm else ""
        btxt     = f"UNAUTHORISED CONSTRUCTION DETECTED  [{unauth_count} structure(s){area_str}]"
        bcol     = (0, 0, 160)
    elif unauth_count == 0:
        verdict  = "AUTHORISED"
        area_str = f" — ~{int(total_auth_sqm)} m²" if total_auth_sqm else ""
        btxt     = f"ALL CONSTRUCTION AUTHORISED  [{auth_count} structure(s){area_str}]"
        bcol     = (0, 130, 0)
    else:
        verdict = "MIXED"
        btxt    = f"MIXED: {auth_count} AUTHORISED | {unauth_count} UNAUTHORISED"
        bcol    = (130, 80, 0)

    # Semi-transparent banner at top
    banner = result_overlay.copy()
    cv2.rectangle(banner, (0, 0), (w, 52), bcol, -1)
    result_overlay = cv2.addWeighted(banner, 0.75, result_overlay, 0.25, 0)
    cv2.putText(result_overlay, btxt, (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255,255,255), 2, cv2.LINE_AA)

    short = "ResNet50" if TORCH_AVAILABLE else "OpenCV Diff"
    cv2.putText(result_overlay, f"Method: {short}",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (220, 220, 220), 1, cv2.LINE_AA)

    # ── Step 9: Save outputs ──────────────────────────────────────────────
    cv2.imwrite(result_path, result_overlay)

    p_combined = result_path.replace(".jpg", "_mask_combined.png")
    p_auth     = result_path.replace(".jpg", "_mask_auth.png")
    p_unauth   = result_path.replace(".jpg", "_mask_unauth.png")
    p_heatmap  = result_path.replace(".jpg", "_deep_change_heatmap.png")
    p_osm      = result_path.replace(".jpg", "_osm_zone.png")

    colour_mask = np.zeros((h, w, 3), dtype=np.uint8)
    colour_mask[mask_auth   > 0] = (0, 200,   0)
    colour_mask[mask_unauth > 0] = (0,   0, 200)
    cv2.imwrite(p_combined, colour_mask)
    cv2.imwrite(p_auth,     mask_auth)
    cv2.imwrite(p_unauth,   mask_unauth)
    cv2.imwrite(p_heatmap,  cv2.applyColorMap(cmap, cv2.COLORMAP_JET))

    if permit_mask is not None:
        osm_vis = af.copy()
        tint    = np.zeros_like(af)
        tint[permit_mask > 0] = (0, 90, 0)
        osm_vis = cv2.addWeighted(tint, 0.45, osm_vis, 0.55, 0)
        cv2.putText(osm_vis, "AUTHORISED ZONE (OSM)", (10, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,100), 2, cv2.LINE_AA)
        cv2.imwrite(p_osm, osm_vis)
    else:
        p_osm = None

    metrics = {
        "verdict":               verdict,
        "total_changes":         total,
        "authorised_count":      auth_count,
        "unauthorised_count":    unauth_count,
        "authorised_area_px":    int(sum(auth_areas_px)),
        "unauthorised_area_px":  int(sum(unauth_areas_px)),
        "authorised_area_sqm":   total_auth_sqm,
        "unauthorised_area_sqm": total_unauth_sqm,
        "buildings":             buildings,
        "change_ratio_percent":  round(float(np.sum(clean>0))/clean.size*100, 2),
        "changed_pixels":        int(np.sum(clean>0)),
        "alignment_status":      align_status,
        "method_used":           method,
        "torch_available":       TORCH_AVAILABLE,
        "error_msg":             None,
        "mask_path":             p_combined,
        "mask_auth_path":        p_auth,
        "mask_unauth_path":      p_unauth,
        "change_map_path":       p_heatmap,
        "osm_overlay_path":      p_osm,
    }
    return result_overlay, metrics
