import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(' ') # select your model.pt path
    model.predict(source=' .mp4',
                  imgsz=640,
                  device='0',
                  project='runs/detect',
                  name='pose_test1',
                  save=True,
                #   visualize=True # visualize model features maps
                )