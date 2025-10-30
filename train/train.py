from ultralytics import YOLO
import torch

# 1. GPU 사용 가능 여부 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# 2. YOLOv8n 모델 로드
model = YOLO('yolov8n.pt')

# 3. 모델 학습
results = model.train(
   #augment default만 있는 버전
   data='./dataset.yaml',
   imgsz=480,
   epochs=600,
   batch=256,
   patience=70,       # 70 epoch 동안 성능 개선이 없으면 학습 조기 종료
   workers=8,
   optimizer='AdamW', # AdamW 옵티마이저 사용   
   project='./runs/train',
   name='model_name', 
   plots=True         # 학습 중 loss, mAP 등의 그래프 생성
)

print(f"Results saved to: '{results.save_dir}'")