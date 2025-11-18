import cv2
import numpy as np
import mediapipe as mp
import time

BG_CHAR = "."
FG_CHARS = np.array(list(" .,:;i1tfLCG08@#W%&$xa=?"))

CHAR_SIZE = 16
THICKNESS = 1
MIN_FONT = 0.5
MAX_FONT = 2.2

SEG_THRESH = 0.15
GLOW_INTENSITY = 1.6
GLOW_BLUR = 20

SAVE_PREFIX = "ascii_frame_"
SAVE_INDEX = 0
PNG_INDEX = 0
REC_INDEX = 0

mp_hands = mp.solutions.hands.Hands(max_num_hands=2)
mp_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(1)


def map_range(v,a,b,c,d):
    return c + ((max(min(v,b),a) - a) / (b - a)) * (d - c)


def save_ascii_txt(matrix, filename):
    with open(filename,"w") as f:
        for r in matrix:
            f.write("".join(r)+"\n")


def save_glow_png_transparent(img, filename):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    alpha = np.where(gray > 10, 255, 0).astype(np.uint8)
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[...,3] = alpha
    cv2.imwrite(filename, bgra)


cap = cv2.VideoCapture(0)

recording = False
writer = None

prev_time  = time.time()
total_time = 0.0
frame_counter = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    now = time.time()
    dt = now - prev_time
    prev_time = now
    total_time += dt
    frame_counter += 1

    h,w,_ = frame.shape
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    seg = mp_seg.process(rgb).segmentation_mask
    seg_mask = (seg > SEG_THRESH).astype(np.uint8)

    hand_mask = np.zeros_like(seg_mask)
    FONT_SCALE = 1.0

    hands = mp_hands.process(rgb)
    if hands.multi_hand_landmarks:
        for lm in hands.multi_hand_landmarks:
            xs, ys = [], []
            for p in lm.landmark:
                xs.append(int(p.x * w))
                ys.append(int(p.y * h))
            for x,y in zip(xs,ys):
                cv2.circle(hand_mask,(x,y),25,1,-1)

        x1,y1 = lm.landmark[4].x, lm.landmark[4].y
        x2,y2 = lm.landmark[8].x, lm.landmark[8].y
        dist = ((x1-x2)**2+(y1-y2)**2)**0.5
        FONT_SCALE = map_range(dist,0.03,0.40,MIN_FONT,MAX_FONT)

    final_mask = np.clip(seg_mask + hand_mask, 0, 1)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray,(w//CHAR_SIZE,h//CHAR_SIZE))
    idx = (small / 255 * (len(FG_CHARS)-1)).astype(int)
    ascii_fg  = FG_CHARS[idx]
    mask_small = cv2.resize(final_mask,(w//CHAR_SIZE,h//CHAR_SIZE))

    canvas = np.zeros((h,w,3),dtype=np.uint8)
    EXPORT_MATRIX = []

    for yy in range(ascii_fg.shape[0]):
        row=[]
        for xx in range(ascii_fg.shape[1]):
            ch = ascii_fg[yy,xx] if mask_small[yy,xx]==1 else BG_CHAR
            row.append(ch)
            cv2.putText(canvas,ch,(xx*CHAR_SIZE,yy*CHAR_SIZE+CHAR_SIZE),
                        cv2.FONT_HERSHEY_PLAIN,
                        FONT_SCALE,(255,255,255),THICKNESS)
        EXPORT_MATRIX.append(row)

    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_canvas,(0,0),GLOW_BLUR)
    glow_color = cv2.applyColorMap(blur, cv2.COLORMAP_PINK)
    glow_mix   = cv2.addWeighted(canvas,1.0,glow_color,GLOW_INTENSITY,0)

    cv2.rectangle(glow_mix,(0,0),(w,45),(0,0,0),-1)
    cv2.putText(glow_mix,"Q Quit | S Save TXT+PNG | R Record",
                (15,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(220,220,255),2)

    if recording:
        cv2.putText(glow_mix,"● REC",(w-110,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,50,255),3)

    cv2.imshow("ASCII REALTIME EFFECT CAM", glow_mix)

    if recording:
        writer.write(glow_mix)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        txt = f"{SAVE_PREFIX}{SAVE_INDEX}.txt"
        png = f"{SAVE_PREFIX}{PNG_INDEX}.png"
        save_ascii_txt(EXPORT_MATRIX,txt)
        save_glow_png_transparent(glow_mix,png)
        SAVE_INDEX += 1
        PNG_INDEX  += 1

    if key == ord("r"):
        if not recording:
            if total_time > 0 and frame_counter > 10:
                fps_est = frame_counter / total_time
            else:
                fps_est = 15.0

            fps_est = max(5.0,min(30.0,fps_est))

            rec_file = f"ascii_record_{REC_INDEX}.mp4"
            writer = cv2.VideoWriter(
                rec_file,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps_est,
                (w,h)
            )
            recording = True
            REC_INDEX += 1
        else:
            recording = False
            writer.release()

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
