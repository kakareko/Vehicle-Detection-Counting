import glob
import os
import random

import cv2

from centroidtracker import CentroidTracker
from yolov5_object_detector import YOLOv5ObjectDetector

CONFIDENCE_THRES = 0.25
VIDEO_PATH = '../../../../videos/*.mp4'

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
           'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
           'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
           'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
           'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
           'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def plot_one_box(x, img, color=(0, 160, 0), cls=None):
    """
    Plots one bounding box on image img

    :param x: tuple or list with coordinates of left top corner and right bottom corner
    :param img:
    :param color:
    :param cls:
    """
    tl = 2
    c1 = (int(x[0]), int(x[1]))
    c2 = (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    label = f'{CLASSES[cls]}'
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

    corner1 = c1[0], c2[1] - t_size[1] - 3
    corner2 = c1[0] + t_size[0], c2[1]

    cv2.rectangle(img, corner1, corner2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(img, label, (corner1[0], corner2[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                lineType=cv2.LINE_AA)


if __name__ == '__main__':
    print('[INFO] loading model...')
    tracked_classes = [2, 5, 7]
    model = YOLOv5ObjectDetector(conf_thres=CONFIDENCE_THRES, classes=tracked_classes)

    os.makedirs('out', exist_ok=True)
    counts_path = 'out/counts.txt'

    for video_path_in in glob.glob(VIDEO_PATH):
        video_path_out = os.path.join('out', os.path.split(video_path_in)[1])

        print(f'[INFO] Processing {video_path_in} -> {video_path_out}')
        vs = cv2.VideoCapture(video_path_in)

        vs_length = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        vs_fps = int(vs.get(cv2.CAP_PROP_FPS))
        W = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = 'avc1'  # output video codec
        writer = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*fourcc), vs_fps, (W, H))

        c_trackers = {}
        colors = {}
        for cl in tracked_classes:
            c_trackers[cl] = CentroidTracker(max_disappeared=5, max_distance=50)
            colors[cl] = (random.randint(0, 128), random.randint(0, 128), random.randint(0, 128))

        total_frames = 0

        while True:
            frame = vs.read()
            frame = frame[1]

            if frame is None:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detections = model.detect(rgb)

            rects = {}
            for cl in tracked_classes:
                rects[cl] = []

            for i in range(detections.shape[0]):
                cl = int(detections[i, 5])
                rects[cl].append(tuple(detections[i, :4]))
                plot_one_box(detections[i, :4], frame, colors[cl], cl)

            for cl in tracked_classes:
                objects = c_trackers[cl].update(rects[cl])

                for (object_id, centroid) in objects.items():
                    text = f'{CLASSES[cl]} {object_id}'
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            writer.write(frame)

            total_frames += 1

            if total_frames % 30 == 0:
                print('frame {} of {} '.format(total_frames, vs_length))

        if total_frames % 30 != 0:
            print('frame {} of {} '.format(total_frames, vs_length))

        s_det = ''
        for cl, tracker in c_trackers.items():
            s_det += f'{CLASSES[cl]}: {tracker.next_object_id} '
        print(s_det)

        vs.release()
        writer.release()

        with open(counts_path, 'a') as f:
            f.write(f'{os.path.split(video_path_in)[1]} - {s_det}\n')
