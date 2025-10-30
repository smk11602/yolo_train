import cv2
import os
from ultralytics import YOLO

# --- 1. 설정 ---
MODEL_PATH = 'model.pt' #수정
VIDEO_PATH = 'video.mp4' #수정
CONF_THRESHOLD = 0.5

video_dir = os.path.dirname(VIDEO_PATH)
video_name_base = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_PATH = os.path.join(video_dir, f"{video_name_base}_yolo-_conf0.5.mp4")

CLASS_DICT = {
    0: 'chickenmayo', 1: 'seaweed_soup', 2: 'condition_stick',
    3: 'pepero_original', 4: 'pulmuone_spring_water', 5: 'samdasoo',
    6: 'creeat_protein_bar'
}

COLOR_MAP = {
    0: (255, 158, 66),   # (R=66, G=158, B=255) -> (B=255, G=158, R=66)
    1: (40, 181, 224),   # (R=224, G=181, B=40) -> (B=40, G=181, R=224)
    2: (209, 247, 84),   # (R=84, G=247, B=209) -> (B=209, G=247, R=84)
    3: (148, 148, 255),   # (R=255, G=148, B=148) -> (B=148, G=148, R=255)
    4: (110, 255, 84),   # (R=84, G=255, B=110) -> (B=110, G=255, R=84)
    5: (235, 213, 47),   # (R=47, G=213, B=235) -> (B=235, G=213, R=47)
    6: (247, 84, 171)    # (R=171, G=84, B=247) -> (B=247, G=84, R=171)
}

# 폰트 및 라인 설정
FONT_SCALE = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_THICKNESS = 2
TEXT_THICKNESS = 1

try:
    model = YOLO(MODEL_PATH)
    print(f"모델 로드 성공: {MODEL_PATH}")
except Exception as e:
    print(f"오류: 모델 로드 실패. {e}")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"오류: 비디오 파일을 열 수 없습니다. {VIDEO_PATH}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
midpoint_y = frame_height / 2 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

print(f"추론 및 '중심점' 필터링 시작... (입력: {VIDEO_PATH})")
print(f"결과 저장 위치: {OUTPUT_PATH}")
print(f"(참고: 프레임 높이 {frame_height}, 필터링 기준 Y={midpoint_y})")

frame_count = 0
print("\n--- 감지 로그 (필터링 통과) ---") 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    detected_in_this_frame = False 
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Bbox의 세로 중심 Y좌표 계산
        center_y = (y1 + y2) / 2
        
        # 중심점이 midpoint_y (프레임 중앙선)보다 위에 있는지 확인
        if center_y < midpoint_y:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            label_name = CLASS_DICT.get(cls_id, f'Class {cls_id}')
            label = f'{label_name}: {conf:.2f}'

            # 필터링을 통과한 객체의 ID와 이름을 콘솔에 출력
            print(f"  [Frame {frame_count:04d}] DETECTED: ID={cls_id}, Name={label_name} (Conf: {conf:.2f})")
            detected_in_this_frame = True # ❗

            color = COLOR_MAP.get(cls_id, (255, 255, 255)) 
            
            # Bbox 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
            
            # 텍스트 배경 계산 및 그리기
            (w, h), _ = cv2.getTextSize(label, FONT, FONT_SCALE, TEXT_THICKNESS)
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + h + 10
            cv2.rectangle(frame, (x1, text_y - h - 4), (x1 + w, text_y), color, -1)

            # 텍스트 (검은색)
            cv2.putText(frame, 
                        label, 
                        (x1, text_y - 3), 
                        FONT, 
                        FONT_SCALE, 
                        (0, 0, 0), 
                        TEXT_THICKNESS, 
                        cv2.LINE_AA)

    out.write(frame)
    
    frame_count += 1
    if not detected_in_this_frame and frame_count % (int(fps) * 5) == 0: # 5초마다
        print(f"  ... (Processing frame {frame_count}) ...")
    elif frame_count % (int(fps) * 5) == 0:
        # 감지 로그가 출력되는 중에는 이 메시지를 생략하여 깔끔하게
        pass

# --- 6. 자원 해제 ---
cap.release()
out.release()

print(f"총 {frame_count} 프레임이 처리되었습니다.")
print(f"결과 비디오가 다음 위치에 저장되었습니다: {OUTPUT_PATH}")