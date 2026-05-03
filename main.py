import cv2
import pickle
import cvzone
import numpy as np
from datetime import datetime

VIDEO_PATH      = 'carPark.mp4'
POSITIONS_FILE  = 'CarParkPos'
WIDTH, HEIGHT   = 107, 48
THRESHOLD       = 900
NIGHT_THRESHOLD = 60      
HEATMAP_ALPHA   = 0.4    

ZONE_NAMES  = ['A', 'B', 'C']
ZONE_COLORS = [(255, 100, 0), (0, 165, 255), (128, 0, 128)]  # BGR

with open(POSITIONS_FILE, 'rb') as f:
    posList = pickle.load(f)

xs = [p[0] for p in posList]
x_min, x_max = min(xs), max(xs)
zone_width = (x_max - x_min + WIDTH) / len(ZONE_NAMES)

def get_zone(x):
    idx = int((x - x_min) / zone_width)
    return min(idx, len(ZONE_NAMES) - 1)

prev_states  = {pos: False for pos in posList}
cur_states   = {pos: False for pos in posList}

occupied_since = {}

total_entries = 0
total_exits   = 0

heatmap_accum = None
frame_count   = 0

cap = cv2.VideoCapture(VIDEO_PATH)

def is_night_mode(frame):
    """Return True if average brightness is below NIGHT_THRESHOLD."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean() < NIGHT_THRESHOLD


def preprocess(frame, night):
    """Return binary image ready for spot counting."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if night:
        gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        thresh = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 21, 10)
    else:
        blur   = cv2.GaussianBlur(gray, (3, 3), 1)
        thresh = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 16)
    median  = cv2.medianBlur(thresh, 5)
    kernel  = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(median, kernel, iterations=1)
    return dilated


def format_duration(seconds):
    """Convert seconds to hh:mm:ss string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h{m:02d}m"
    elif m:
        return f"{m}m{s:02d}s"
    else:
        return f"{s}s"


def update_heatmap(accum, processed_frame):
    """Add current occupied pixels to heatmap accumulator."""
    accum += processed_frame.astype(np.float32)


def render_heatmap(accum, frame_count):
    """Convert accumulator to a coloured heatmap image."""
    if frame_count == 0:
        return None
    norm = accum / frame_count
    norm_uint8 = (norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)
    return heatmap


def check_parking_space(img, imgPro, night):
    global total_entries, total_exits, frame_count, heatmap_accum

    if heatmap_accum is None:
        heatmap_accum = np.zeros(imgPro.shape, dtype=np.float32)

    update_heatmap(heatmap_accum, imgPro)
    frame_count += 1

    now = datetime.now()

    zone_free  = [0] * len(ZONE_NAMES)
    zone_total = [0] * len(ZONE_NAMES)

    for pos in posList:
        x, y    = pos
        zone_id = get_zone(x)
        zone_total[zone_id] += 1

        crop  = imgPro[y:y + HEIGHT, x:x + WIDTH]
        count = cv2.countNonZero(crop)

        occupied = count >= THRESHOLD
        cur_states[pos] = occupied

        was_occupied = prev_states[pos]
        if not was_occupied and occupied:
            total_entries += 1
            occupied_since[pos] = now
        elif was_occupied and not occupied:
            
            total_exits += 1
            occupied_since.pop(pos, None)

       duration_str = ""
        if occupied and pos in occupied_since:
            elapsed      = (now - occupied_since[pos]).total_seconds()
            duration_str = format_duration(elapsed)

        zone_color = ZONE_COLORS[zone_id]
        if not occupied:
            color     = (0, 255, 0)
            thickness = 5
            zone_free[zone_id] += 1
        else:
            color     = (0, 0, 255)
            thickness = 2

        cv2.rectangle(img, pos, (pos[0] + WIDTH, pos[1] + HEIGHT), color, thickness)

        cv2.putText(img, ZONE_NAMES[zone_id], (x + 2, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, zone_color, 1)

        cvzone.putTextRect(img, str(count), (x, y + HEIGHT - 3),
                           scale=1, thickness=2, offset=0, colorR=color)

        if duration_str:
            cvzone.putTextRect(img, duration_str, (x + WIDTH + 4, y + HEIGHT // 2 - 10),
                               scale=1.2, thickness=2, offset=4, colorR=(0, 200, 255))

    prev_states.update(cur_states)

    return zone_free, zone_total


def draw_hud(img, zone_free, zone_total, night, fps):
    h, w = img.shape[:2]

    total_free  = sum(zone_free)
    total_spots = sum(zone_total)

    items = []
    items.append((f'Free: {total_free}/{total_spots}', (0, 180, 0)))
    for i, name in enumerate(ZONE_NAMES):
        items.append((f'{name}:{zone_free[i]}/{zone_total[i]}', ZONE_COLORS[i]))
    items.append((f'In:{total_entries}', (0, 220, 220)))
    items.append((f'Out:{total_exits}', (0, 180, 255)))
    if night:
        items.append(('NIGHT', (0, 100, 255)))
    items.append((f'FPS:{fps:.0f}', (120, 120, 120)))

    x_cursor = 10
    y_pos    = 35
    for text, color in items:
        cvzone.putTextRect(img, text, (x_cursor, y_pos),
                           scale=1.1, thickness=2, offset=6, colorR=color)
        x_cursor += len(text) * 15 + 20

#  MAIN LOOP
show_heatmap = False
prev_time    = datetime.now()

print("Controls:  Q = quit  |  H = toggle heatmap overlay")

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    success, img = cap.read()
    if not success:
        break


    now      = datetime.now()
    fps      = 1.0 / max((now - prev_time).total_seconds(), 1e-6)
    prev_time = now

    night   = is_night_mode(img)
    imgPro  = preprocess(img, night)

    zone_free, zone_total = check_parking_space(img, imgPro, night)

    draw_hud(img, zone_free, zone_total, night, fps)

    # Heatmap overlay (toggle with H)
    if show_heatmap and heatmap_accum is not None:
        heatmap_img = render_heatmap(heatmap_accum, frame_count)
        if heatmap_img is not None:
            img = cv2.addWeighted(img, 1 - HEATMAP_ALPHA,
                                  heatmap_img, HEATMAP_ALPHA, 0)

    cv2.imshow("Parking System", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):
        show_heatmap = not show_heatmap
        print(f"Heatmap overlay: {'ON' if show_heatmap else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
