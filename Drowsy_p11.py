import cv2
import mediapipe as mp
import math
import winsound

# ---------------- INIT ----------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

# ---------------- CALIBRATION ----------------
calibration_frames = 60
frame_count = 0

base_angle = 0
base_ear = 0
base_chin_y = 0

# ---------------- TRACKING ----------------
eye_counter = 0
final_score = 0

prev_chin_y = 0

# -------- NOD DETECTION --------
nod_stage = 0   # 0 = neutral, 1 = down
nod_count = 0

# -------- SIDE TILT --------
tilt_state = 0
tilt_cycles = 0

# ---------------- DROWSY HOLD ----------------
drowsy_timer = 0
DROWSY_HOLD_FRAMES = 90

# ---------------- SOUND ----------------
beep_counter = 0

# ---------------- SETTINGS ----------------
EAR_DROP_RATIO = 0.7
ANGLE_THRESHOLD = 10
DROWSY_SCORE = 8

# ---------------- FUNCTIONS ----------------
def calculate_ear(eye):
    A = math.dist(eye[1], eye[5])
    B = math.dist(eye[2], eye[4])
    C = math.dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb)

    h, w, _ = frame.shape
    score = 0

    if result.multi_face_landmarks:
        face = result.multi_face_landmarks[0]

        # -------- HEAD --------
        forehead = face.landmark[10]
        chin = face.landmark[152]

        f_pt = (int(forehead.x * w), int(forehead.y * h))
        c_pt = (int(chin.x * w), int(chin.y * h))

        dx = c_pt[0] - f_pt[0]
        dy = c_pt[1] - f_pt[1]

        angle = abs(math.degrees(math.atan2(dx, dy)))

        # -------- EYES --------
        left_eye, right_eye = [], []

        for i in [33,160,158,133,153,144]:
            pt = face.landmark[i]
            left_eye.append((int(pt.x*w), int(pt.y*h)))

        for i in [362,385,387,263,373,380]:
            pt = face.landmark[i]
            right_eye.append((int(pt.x*w), int(pt.y*h)))

        ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2

        # -------- CALIBRATION --------
        frame_count += 1

        if frame_count < calibration_frames:
            base_angle += angle
            base_ear += ear
            base_chin_y += c_pt[1]

            cv2.putText(frame, "CALIBRATING...", (20,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            continue

        if frame_count == calibration_frames:
            base_angle /= calibration_frames
            base_ear /= calibration_frames
            base_chin_y /= calibration_frames
            prev_chin_y = base_chin_y

        ear_threshold = base_ear * EAR_DROP_RATIO

        # -------- EYE DETECTION --------
        if ear < ear_threshold:
            eye_counter += 1
        else:
            eye_counter = 0

        if eye_counter > 8:
            score += 3

        if ear < ear_threshold * 0.75:
            score += 3

        # -------- HEAD TILT --------
        if abs(angle - base_angle) > ANGLE_THRESHOLD:
            score += 1

        # -------- ✅ NOD DETECTION (FIXED) --------

        chin_offset = c_pt[1] - base_chin_y

        # Stage 1: head goes DOWN
        if chin_offset > 18:
            if nod_stage == 0:
                nod_stage = 1

        # Stage 2: head comes BACK UP
        elif chin_offset < 8:
            if nod_stage == 1:
                nod_count += 1
                nod_stage = 0

        # Detect repeated nodding
        if nod_count >= 2:
            score += 4
            nod_count = 0

        # FAST DROP (instant nod)
        delta = c_pt[1] - prev_chin_y
        if delta > 7:
            score += 3

        prev_chin_y = c_pt[1]

        # -------- SIDE TILT --------
        if dx > 15:
            if tilt_state == 0:
                tilt_state = 1

        elif dx < -15:
            if tilt_state == 0:
                tilt_state = -1

        if tilt_state == 1 and dx < -10:
            tilt_cycles += 1
            tilt_state = -1

        elif tilt_state == -1 and dx > 10:
            tilt_cycles += 1
            tilt_state = 1

        if tilt_cycles >= 2:
            score += 3
            tilt_cycles = 0

        # -------- STABILITY --------
        if eye_counter == 0 and abs(angle - base_angle) < 8:
            final_score = max(0, final_score - 4)

        # -------- DEBUG --------
        cv2.putText(frame, f'EAR: {round(ear,2)}', (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        cv2.putText(frame, f'Angle: {int(angle)}', (20,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    else:
        final_score = max(0, final_score - 3)

    # -------- MEMORY --------
    final_score = max(0, final_score - 3)
    final_score += score

    # -------- DROWSY HOLD --------
    if final_score >= DROWSY_SCORE:
        drowsy_timer = DROWSY_HOLD_FRAMES

    if drowsy_timer > 0:
        status = "DROWSY"
        color = (0,0,255)

        beep_counter += 1
        if beep_counter % 20 == 0:
            winsound.Beep(1200, 200)

        drowsy_timer -= 1
    else:
        status = "NORMAL"
        color = (0,255,0)
        beep_counter = 0

    # -------- COUNTDOWN --------
    if drowsy_timer > 0:
        seconds_left = round(drowsy_timer / 30, 1)
        cv2.putText(frame, f'Alert: {seconds_left}s', (20,200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # -------- STATUS --------
    cv2.putText(frame, status, (20,160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    cv2.imshow("Drowsiness Detection FINAL FIXED", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()