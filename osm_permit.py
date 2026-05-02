"""
osm_permit.py  — v3  (multi-endpoint, retry-aware)
─────────────────────────────────────────────────────────────────────────────
Fetches OSM building footprints via Overpass API.

FIX in v3:
  • 4 Overpass API endpoints tried in order — if one times out or returns
    a 504/429, the next one is tried automatically.
  • Per-endpoint timeout of 25 seconds, total max wait ~100 seconds.
  • Clear status message tells user which server responded.
"""

import requests
import numpy as np
import cv2

# ── Multiple Overpass endpoints (tried in order) ──────────────────────────
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",          # primary
    "https://overpass.kumi.systems/api/interpreter",    # EU mirror
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",  # Russia CDN
    "https://overpass.openstreetmap.ru/api/interpreter", # OSM.ru
]


def fetch_osm_buildings(lat_min, lon_min, lat_max, lon_max, timeout=25):
    """
    Try each Overpass endpoint in turn until one responds successfully.
    Returns (json_data, endpoint_used) or raises RuntimeError if all fail.
    """
    query = f"""
[out:json][timeout:{timeout}];
(
  way["building"]({lat_min},{lon_min},{lat_max},{lon_max});
  relation["building:part"]({lat_min},{lon_min},{lat_max},{lon_max});
);
out geom;
"""
    last_error = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            resp = requests.post(
                endpoint,
                data={"data": query},
                timeout=timeout + 5,
                headers={"User-Agent": "ConstructionDetectionSystem/1.0"}
            )
            resp.raise_for_status()
            return resp.json(), endpoint
        except requests.RequestException as e:
            last_error = e
            print(f"[OSM] {endpoint} failed: {e} — trying next...")
            continue

    raise RuntimeError(
        f"All Overpass API endpoints failed. Last error: {last_error}"
    )


# ── Coordinate conversion ─────────────────────────────────────────────────

def latlon_to_pixel(lat, lon, lat_min, lon_min, lat_max, lon_max,
                    img_w, img_h):
    x = (lon - lon_min) / (lon_max - lon_min) * img_w
    y = (lat_max - lat) / (lat_max - lat_min) * img_h
    return (max(0, min(img_w - 1, int(x))),
            max(0, min(img_h - 1, int(y))))


# ── Mask generation ───────────────────────────────────────────────────────

def create_osm_permit_mask(lat_min, lon_min, lat_max, lon_max,
                            img_w, img_h, osm_data, buffer_px=20):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    count = 0

    for element in osm_data.get("elements", []):
        if element.get("type") != "way":
            continue
        geometry = element.get("geometry", [])
        if len(geometry) < 3:
            continue
        pts = [list(latlon_to_pixel(
                    n["lat"], n["lon"],
                    lat_min, lon_min, lat_max, lon_max,
                    img_w, img_h))
               for n in geometry]
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
        count += 1

    if buffer_px > 0 and count > 0:
        k = np.ones((buffer_px * 2 + 1, buffer_px * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, k, iterations=1)

    return mask, count


# ── Debug overlay ─────────────────────────────────────────────────────────

def create_debug_overlay(after_img, permit_mask, osm_data,
                          lat_min, lon_min, lat_max, lon_max):
    h, w    = after_img.shape[:2]
    overlay = after_img.copy()

    tint = np.zeros_like(after_img)
    tint[permit_mask > 0] = (0, 80, 0)
    overlay = cv2.addWeighted(tint, 0.40, overlay, 0.60, 0)

    for element in osm_data.get("elements", []):
        if element.get("type") != "way":
            continue
        geometry = element.get("geometry", [])
        if len(geometry) < 3:
            continue
        pts = [list(latlon_to_pixel(
                    n["lat"], n["lon"],
                    lat_min, lon_min, lat_max, lon_max, w, h))
               for n in geometry]
        cv2.polylines(overlay,
                      [np.array(pts, dtype=np.int32)],
                      True, (0, 255, 255), 1)

    cv2.putText(overlay, "OSM BUILDINGS (Yellow = polygon edges)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return overlay


# ── Main wrapper ──────────────────────────────────────────────────────────

def build_permit_mask(lat_min, lon_min, lat_max, lon_max,
                       img_w, img_h,
                       save_path=None,
                       after_img=None,
                       debug_overlay_path=None):
    """
    Fetch OSM buildings → rasterise → return permit mask.

    Returns
    -------
    mask           : uint8 binary mask (255 = authorised zone)
    building_count : int
    status_msg     : human-readable string for the UI
    osm_data       : raw OSM JSON dict
    """
    try:
        osm_data, endpoint_used = fetch_osm_buildings(
            lat_min, lon_min, lat_max, lon_max
        )
        mask, count = create_osm_permit_mask(
            lat_min, lon_min, lat_max, lon_max,
            img_w, img_h, osm_data,
            buffer_px=20
        )

        server_name = endpoint_used.split("/")[2]   # e.g. "overpass-api.de"

        if count == 0:
            status_msg = (
                f"No OSM buildings found in this bounding box "
                f"(via {server_name}). "
                "All changes flagged UNAUTHORISED. "
                "Check your coordinates or try a more urbanised area."
            )
        else:
            covered_pct = round(np.sum(mask > 0) / mask.size * 100, 1)
            status_msg  = (
                f"✅ {count} OSM building footprint(s) loaded via {server_name} — "
                f"covering {covered_pct}% of image area as authorised zone."
            )

    except Exception as e:
        mask       = np.zeros((img_h, img_w), dtype=np.uint8)
        count      = 0
        osm_data   = {"elements": []}
        status_msg = (
            f"⚠️ All OSM servers failed ({e}). "
            "All changes flagged UNAUTHORISED. "
            "Check your internet connection and try again."
        )

    if save_path:
        cv2.imwrite(save_path, mask)

    if after_img is not None and debug_overlay_path and count > 0:
        debug = create_debug_overlay(
            after_img, mask, osm_data,
            lat_min, lon_min, lat_max, lon_max
        )
        cv2.imwrite(debug_overlay_path, debug)

    return mask, count, status_msg, osm_data
