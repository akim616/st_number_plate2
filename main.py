from io import StringIO
from pathlib import Path
import streamlit as st
import time
from detect import detect
import os
import sys
import argparse
from PIL import Image
import numpy as np
import cv2


def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)


def read_txt_boundries(filepath):
    f = open(filepath, 'r')
    lines = f.readlines()

    # read objects from each line of .txt
    objects = []
    for line in lines:
        line=line.rstrip()
        obj = [int(float(i)) for i in line.split(' ')]
        objects.append(obj)

    x1 = objects[0][1] - (objects[0][3] /2)  # xmin
    y1 = objects[0][2] - (objects[0][4] /2)  # ymin
    x2 = x1 + objects[0][3] # xmax
    y2 = y1 + objects[0][4] # ymax
    return int(x1), int(y1), int(x2), int(y2)


def blurred_img(img_path, coors):
    img=cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2=coors
    cnt= np.array(
        [[[x1, y1]],
        [[x2, y1]],
        [[x2, y2]],
        [[x1, y2]]], dtype='int32')
    out = img.copy()
    blurred_part = cv2.blur(out[ y1: y2, x1: x2 ], ksize=(20, 20) )
    blurred = out.copy()
    blurred[y1: y2, x1: x2] = blurred_part
    return blurred


if __name__ == '__main__':

    st.title('YOLOv5 Streamlit App')

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true', default=True,
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    source = ("Image",)
    source_index = st.sidebar.selectbox("Select Input", range(
        len(source)), format_func=lambda x: source[x])

    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "Upload", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='Loading...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'data/images/{uploaded_file.name}')
                opt.source = f'data/images/{uploaded_file.name}'
        else:
            is_valid = False
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='Loading...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                opt.source = f'data/videos/{uploaded_file.name}'
        else:
            is_valid = False

    if is_valid:
        print('valid')
        if st.button('Detect'):

            detect(opt)

            if source_index == 0:
                with st.spinner(text='Preparing Images'):
                    for img in os.listdir(get_detection_folder()):
                            pass
                    
                    if len(os.listdir(os.path.join(get_detection_folder().split('\\')[-1], 'labels')))!=0:
                        for t in os.listdir(os.path.join('runs','detect', get_detection_folder().split('\\')[-1], 'labels')):
                            st.write(read_txt_boundries(str(Path(f'{get_detection_folder()}') /'labels'/ t)))
                            st.image(blurred_img(opt.source, read_txt_boundries(str(Path(f'{get_detection_folder()}') /'labels'/ t))), width=600)
                    else:
                        st.image(str(Path(f'{get_detection_folder()}') / img), width=600)
            else:
                with st.spinner(text='Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        st.video(str(Path(f'{get_detection_folder()}') / vid))

#                     st.balloons()
