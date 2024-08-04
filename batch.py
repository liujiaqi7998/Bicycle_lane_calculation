import os
from pathlib import Path
import glob

import cv2
import pandas as pd
from loguru import logger
from ultralytics import YOLO

# 正确获取当前运行目录
BASE_DIR = Path(__file__).resolve().parent
# 加载路面坑洞模型
Segmentation_pt_Path = os.path.join(BASE_DIR, 'Segmentation', 'runs', 'segment', 'train', 'weights', 'best.pt')
logger.info(f"程序运行目录:{BASE_DIR}")

Segmentation_bestModel = YOLO(Segmentation_pt_Path)
conf_threshold = 0.25
iou_threshold = 0.45


def main():
    logger.info("批量上传启动成功")
    input_dir = os.path.join(BASE_DIR, "图片批量处理", "输入图像")
    output_dir = os.path.join(BASE_DIR, "图片批量处理", "输出图片")
    # 支持的图片扩展名
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    # 存储文件名和预测数量的列表
    results_data = []

    # 遍历目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_extensions):
            # 构建文件路径
            file_path = os.path.join(input_dir, filename)

            # 使用 OpenCV 加载图片
            img = cv2.imread(file_path)
            if img is not None:
                # 将 BGR 格式转换为 RGB 格式
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = Segmentation_bestModel.predict(
                    source=img,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    show_labels=True,
                    show_conf=True,
                )
                results_pic = results[0].plot() if results else None
                # 将处理后的图片保存到 output_dir
                output_path = os.path.join(output_dir, filename)
                if results_pic is not None:
                    results_pic = cv2.cvtColor(results_pic, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_path, results_pic)

                # 记录文件名和预测数量
                results_data.append({
                    "filename": filename,
                    "predict_count": len(results[0].boxes)
                })

                logger.success(f"在 {filename} 处理了 {len(results[0].boxes)} 点位")

    # 生成 CSV 文件
    csv_path = os.path.join(BASE_DIR, "图片批量处理", "log.csv")
    df = pd.DataFrame(results_data)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.success(f"CSV 文件已生成: {csv_path}")


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()
