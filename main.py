import sys
import os
import argparse
from PIL import Image as im
import numpy as np
from word_detector import detect, prepare_img, sort_multiline
from path import Path
import matplotlib.pyplot as plt
import cv2
from typing import List

list_img_names_serial = []

def process_single_image(image_path: Path, parsed):
    """Process a single image file."""
    print(f'Processing file {image_path}')
    img = prepare_img(cv2.imread(image_path), parsed.img_height)
    detections = detect(img,
                        kernel_size=parsed.kernel_size,
                        sigma=parsed.sigma,
                        theta=parsed.theta,
                        min_area=parsed.min_area)
    
    lines = sort_multiline(detections)
    
    plt.imshow(img, cmap='gray')
    num_colors = 7
    colors = plt.cm.get_cmap('rainbow', num_colors)
    for line_idx, line in enumerate(lines):
        for word_idx, det in enumerate(line):
            xs = [det.bbox.x, det.bbox.x, det.bbox.x + det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
            ys = [det.bbox.y, det.bbox.y + det.bbox.h, det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
            plt.plot(xs, ys, c=colors(line_idx % num_colors))
            plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')
            print(det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h)
            crop_img = img[det.bbox.y:det.bbox.y + det.bbox.h, det.bbox.x:det.bbox.x + det.bbox.w]
            
            path = 'test_images2'
            if not os.path.exists(path):
                os.mkdir(path)
                print("Directory Created")

            crop_image_path = os.path.join(path, f"line{line_idx}_word{word_idx}.jpg")
            cv2.imwrite(crop_image_path, crop_img)
            list_img_names_serial.append(crop_image_path)
            print(list_img_names_serial)
    
    with open("img_names_sequence.txt", "w") as textfile:
        for element in list_img_names_serial:
            textfile.write(element + "\n")
    
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, default=Path('uploads/testv6.jpg'))
parser.add_argument('--kernel_size', type=int, default=25)
parser.add_argument('--sigma', type=float, default=11)
parser.add_argument('--theta', type=float, default=7)
parser.add_argument('--min_area', type=int, default=100)
parser.add_argument('--img_height', type=int, default=1000)
parsed = parser.parse_args()

process_single_image(parsed.data, parsed)
