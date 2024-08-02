from pathlib import Path
from loguru import logger
import os
from ultralytics import YOLO


# 正确获取当前运行目录
BASE_DIR = Path(__file__).resolve().parent
trainImagePath = os.path.join(BASE_DIR, 'train', 'images')
yamlFilePath = os.path.join(BASE_DIR, 'data.yaml')

logger.info(f"程序运行目录:{BASE_DIR}")
logger.info(f"训练图像目录:{trainImagePath}")
logger.info(f"data.yaml目录:{yamlFilePath}")

YoloV8_module = 'yolov8n-seg.pt'


def main():
    # 加载seg模型
    model = YOLO(YoloV8_module)
    logger.info(f"加载{YoloV8_module}模型完成")
    # yolo task=detect mode=train model=yolov8n-seg.pt data=data.yaml epochs=30 imgsz=640 workers=4 batch=32
    results = model.train(
        data=yamlFilePath,
        epochs=30,
        imgsz=640,
        batch=32,
        lr0=0.0001,  # 初始学习率
        lrf=0.01,  # 最终学习率 (lr0 * lrf)
        dropout=0.25,  # 使用 dropout 正则化
        device=1,  # 运行的设备，即 cuda device=0
        seed=42,
        workers=10 # 开启高线程数
    )
    logger.success(f"{results.task}训练成功，生成模型在:{os.path.join(BASE_DIR, 'runs', 'segment', 'train', 'weights')}")
    pass


# 按装订区域中的绿色按钮以运行训练
if __name__ == '__main__':
    main()
