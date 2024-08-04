import os
from pathlib import Path

from loguru import logger
from ultralytics import YOLO
import gradio as gr
import numpy as np
import cv2 as cv2
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


def fixColor(image):
    return (cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# 正确获取当前运行目录
BASE_DIR = Path(__file__).resolve().parent
# 加载路面坑洞模型
Segmentation_pt_Path = os.path.join(BASE_DIR, 'Segmentation', 'runs', 'segment', 'train', 'weights', 'best.pt')
logger.info(f"程序运行目录:{BASE_DIR}")

Segmentation_bestModel = YOLO(Segmentation_pt_Path)



def crop_bottom_half(image):

    # 获取图片的高度和宽度
    height, width = image.shape[:2]

    # 计算中线位置
    mid_line = height // 2

    # 裁切并保留下半部分
    bottom_half = image[mid_line:height, :]

    return bottom_half

def find_two_most_distant_lines(lines):
    if lines is None or len(lines) < 2:
        raise ValueError("至少需要两条线")

    max_distance = 0
    line_pair = None

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x1_i, y1_i, x2_i, y2_i = lines[i][0]
            x1_j, y1_j, x2_j, y2_j = lines[j][0]

            # 计算线条 i 和 j 的中心点
            center_i_x = (x1_i + x2_i) / 2
            center_j_x = (x1_j + x2_j) / 2

            # 计算横向距离
            distance = abs(center_i_x - center_j_x)

            if distance > max_distance:
                max_distance = distance
                line_pair = (lines[i][0], lines[j][0])

    return line_pair, max_distance

def line_length(line):
    x1, y1, x2, y2 = line
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def find_longest_lines(line_pair):
    line1, line2 = line_pair
    length1 = line_length(line1)
    length2 = line_length(line2)
    return (line1, length1), (line2, length2)

def calculate_distance_between_lines(line1, line2):
    def point_line_distance(px, py, x1, y1, x2, y2):
        norm = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return abs((py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)) / norm

    x1_1, y1_1, x2_1, y2_1 = line1
    x1_2, y1_2, x2_2, y2_2 = line2

    # 计算每个端点到另一条线的距离
    distances = [
        point_line_distance(x1_1, y1_1, x1_2, y1_2, x2_2, y2_2),
        point_line_distance(x2_1, y2_1, x1_2, y1_2, x2_2, y2_2),
        point_line_distance(x1_2, y1_2, x1_1, y1_1, x2_1, y2_1),
        point_line_distance(x2_2, y2_2, x1_1, y1_1, x2_1, y2_1)
    ]

    return min(distances)

def predict_image(pic):
    # 将 Gradio Image 转换为 PIL Image

    # 将 PIL Image 转换为 NumPy 数组
    numpy_image = np.array(pic)

    # 将 RGB 转换为 BGR，因为 OpenCV 使用 BGR 格式
    image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    image = crop_bottom_half(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    ret, thresh = cv2.threshold(gray, 165, 1000, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 30,  minLineLength=100, maxLineGap=200)

    # 拷贝帧图片
    dmy = image[:, :, 0].copy()
    max_distance = -1
    distance_between_lines = -1
    if lines is not None:
        line_pair, max_distance = find_two_most_distant_lines(lines)
        longest_lines = find_longest_lines(line_pair)
        # 计算两条线之间的距离
        distance_between_lines = calculate_distance_between_lines(longest_lines[0][0], longest_lines[1][0])
        logger.info(f"横向相隔最远的两条线之间的距离: {max_distance}")
        logger.info(f"纵向最长的两条线之间的最小距离: {distance_between_lines}")

        # 画出找到的两条线
        for line, length in longest_lines:
            x1, y1, x2, y2 = line
            cv2.line(dmy, (x1, y1), (x2, y2), (255, 0, 0), 3)




    return dmy,max_distance,distance_between_lines


def main():
    # 接口创建函数
    iface = gr.Interface(
        fn=predict_image,
        inputs=[
            gr.Image(type="pil", label="上传图片"),
        ],
        outputs=[gr.Image(type="pil", label="输出图片"),gr.Text(label="横向相隔最远的两条线之间的距离"),gr.Text(label="纵向最长的两条线之间的最小距离")],
        title="自行车路线",
        description="",
    )

    iface.launch(server_port=25680)


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()
