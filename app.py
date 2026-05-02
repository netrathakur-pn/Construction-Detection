from flask import Flask, render_template, request
import os
import cv2
import traceback
from detection  import detect_changes
from osm_permit import build_permit_mask

app = Flask(__name__)
UPLOAD = 'static/uploads'
RESULT = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD
app.config['RESULT_FOLDER'] = RESULT
os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(RESULT, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_images():

    # ── 1. Validate files ─────────────────────────────────────────────────
    before_file = request.files.get('before')
    after_file  = request.files.get('after')

    if not before_file or before_file.filename == '':
        return render_template('index.html',
                               error="Please upload the Before image.")
    if not after_file or after_file.filename == '':
        return render_template('index.html',
                               error="Please upload the After image.")

    # ── 2. Validate GPS coords ────────────────────────────────────────────
    try:
        lat_max = float(request.form['lat_max'])
        lat_min = float(request.form['lat_min'])
        lon_min = float(request.form['lon_min'])
        lon_max = float(request.form['lon_max'])
    except (KeyError, ValueError):
        return render_template('index.html',
                               error="Please fill in all four GPS coordinate fields.")

    if lat_min >= lat_max:
        return render_template('index.html',
                               error=f"North latitude ({lat_max}) must be greater than South latitude ({lat_min}).")
    if lon_min >= lon_max:
        return render_template('index.html',
                               error=f"East longitude ({lon_max}) must be greater than West longitude ({lon_min}).")
    if not (-90 <= lat_min and lat_max <= 90):
        return render_template('index.html',
                               error="Latitude values must be between -90 and 90.")
    if not (-180 <= lon_min and lon_max <= 180):
        return render_template('index.html',
                               error="Longitude values must be between -180 and 180.")

    # ── 3. Save uploaded images ───────────────────────────────────────────
    before_path = os.path.join(UPLOAD, 'before.jpg')
    after_path  = os.path.join(UPLOAD, 'after.jpg')

    try:
        before_file.save(before_path)
        after_file.save(after_path)
    except Exception as e:
        return render_template('index.html',
                               error=f"Could not save uploaded files: {e}")

    # ── 4. Read and normalise images ──────────────────────────────────────
    img1 = cv2.imread(before_path)
    img2 = cv2.imread(after_path)

    if img1 is None:
        return render_template('index.html',
                               error="Could not read the Before image. Make sure it is a valid JPG or PNG file.")
    if img2 is None:
        return render_template('index.html',
                               error="Could not read the After image. Make sure it is a valid JPG or PNG file.")

    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))
    cv2.imwrite(before_path, img1)
    cv2.imwrite(after_path,  img2)

    # ── 5. Fetch OSM permit mask ──────────────────────────────────────────
    osm_mask_path  = os.path.join(RESULT, 'osm_permit_mask.png')
    osm_debug_path = os.path.join(RESULT, 'osm_debug_overlay.png')

    try:
        permit_mask, osm_count, osm_status, osm_data = build_permit_mask(
            lat_min, lon_min, lat_max, lon_max,
            img_w=w, img_h=h,
            save_path=osm_mask_path,
            after_img=img2,
            debug_overlay_path=osm_debug_path,
        )
    except Exception as e:
        # OSM fetch failed — continue without permit mask (all = UNAUTHORISED)
        print(f"[OSM ERROR] {traceback.format_exc()}")
        permit_mask    = None
        osm_count      = 0
        osm_status     = f"⚠️ OSM fetch failed ({e}). All changes flagged UNAUTHORISED."
        osm_debug_path = None

    if osm_count == 0:
        permit_mask    = None
        osm_debug_path = None

    # ── 6. Run change detection ───────────────────────────────────────────
    result_path = os.path.join(RESULT, 'result.jpg')

    try:
        _, metrics = detect_changes(
            before_path, after_path, result_path,
            permit_mask = permit_mask,
            lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max,
        )
    except Exception as e:
        print(f"[DETECTION ERROR] {traceback.format_exc()}")
        return render_template('index.html',
                               error=f"Detection error: {e}. Check the terminal for details.")

    # ── 7. Handle misalignment error ─────────────────────────────────────
    if metrics.get('verdict') == 'ERROR':
        return render_template('index.html', error=metrics.get('error_msg',
                               "Images could not be compared. Re-export from the same GEP view."))

    # ── 8. Render results ─────────────────────────────────────────────────
    return render_template(
        'result.html',

        # Input images
        before_img         = before_path,
        after_img          = after_path,

        # Output images
        result_img         = result_path,
        osm_mask_img       = osm_mask_path   if osm_count > 0 else None,
        osm_overlay_img    = metrics.get('osm_overlay_path'),
        osm_debug_img      = osm_debug_path,
        mask_img           = metrics['mask_path'],
        mask_auth_img      = metrics['mask_auth_path'],
        mask_unauth_img    = metrics['mask_unauth_path'],
        change_map_img     = metrics.get('change_map_path'),

        # Verdict
        verdict            = metrics['verdict'],

        # Counts
        total_changes      = metrics['total_changes'],
        authorised_count   = metrics['authorised_count'],
        unauthorised_count = metrics['unauthorised_count'],

        # Areas (pixels)
        authorised_area    = metrics['authorised_area_px'],
        unauthorised_area  = metrics['unauthorised_area_px'],

        # Areas (square metres — None if no GPS)
        authorised_sqm     = metrics['authorised_area_sqm'],
        unauthorised_sqm   = metrics['unauthorised_area_sqm'],

        # Per-building detail table
        buildings          = metrics['buildings'],

        # Image-level stats
        change_ratio       = metrics['change_ratio_percent'],
        changed_pixels     = metrics['changed_pixels'],

        # Diagnostics
        method_used        = metrics['method_used'],
        alignment_status   = metrics['alignment_status'],
        torch_available    = metrics['torch_available'],

        # OSM info
        osm_building_count = osm_count,
        osm_status         = osm_status,
        permit_provided    = osm_count > 0,

        # Coordinates
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max,
    )


if __name__ == '__main__':
    app.run(debug=True)
