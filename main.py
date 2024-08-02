import os
from pathlib import Path

import cv2
from loguru import logger
from ultralytics import YOLO
import gradio as gr

# 正确获取当前运行目录
BASE_DIR = Path(__file__).resolve().parent
# 加载路面坑洞模型
Segmentation_pt_Path = os.path.join(BASE_DIR, 'Segmentation', 'runs', 'segment', 'train', 'weights','best.pt')
logger.info(f"程序运行目录:{BASE_DIR}")

Segmentation_bestModel = YOLO(Segmentation_pt_Path)


def predict_image(img, conf_threshold, iou_threshold):
    results = Segmentation_bestModel.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
    )
    return results[0].plot() if results else None

def main():
    # 接口创建函数
    iface = gr.Interface(
        fn=predict_image,
        inputs=[
            gr.Image(type="pil", label="上传图片"),
            gr.Slider(minimum=0, maximum=1, value=0.25, label="可信度阈值"),
            gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU阈值"),
        ],
        outputs=gr.Image(type="pil", label="输出图片"),
        title="路面检测",
        description="基于 yolov8n 和 opencv 实现的自行车道路测算系统",
    )

    iface.launch(server_port=25680)


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()


