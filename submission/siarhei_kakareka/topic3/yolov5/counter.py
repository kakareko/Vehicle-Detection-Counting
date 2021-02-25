import glob
import os
import random

import cv2

from centroidtracker import CentroidTracker
from yolov5_object_detector import YOLOv5ObjectDetector

CONFIDENCE_THRES = 0.25
VIDEO_PATH = '../../../../videos/Topic23-video1.mp4'

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


class NorthBoundController:
    def __init__(self, border, margin):
        """
        Detection of trajectory that went north

        :param border: (x1, y1, x2, y2) - border line from x1, y1 to x2, y2, that is controlled for crossing
        :param margin: - distance in pixels around the border that has to be passed to count the crossing
        """
        self.border = border
        self.margin = margin

        x1, y1, x2, y2 = border
        self.a = (y2-y1)/(x2-x1)
        self.b = (y1*x2-y2*x1)/(x2-x1)

        self.first_detected_below = False
        self.detected_above = False

    def update(self, centroid):
        x, y = centroid

        x1, y1, x2, y2 = self.border
        m = self.margin

        y_border = x * self.a + self.b

        if x1 - m < x < x2 + m:
            if y_border + m < y:               # below the line
                if not self.detected_above:
                    self.first_detected_below = True
            if y_border - m > y:               # above the line
                self.detected_above = True

    @property
    def passed(self):
        return self.first_detected_below and self.detected_above


class RightTurnController:
    def __init__(self, border1, border2):
        self.border1 = border1
        self.border2 = border2

        x1, y1, x2, y2 = border1
        self.border1_a = (x2-x1)/(y2-y1)
        self.border1_b = (x1*y2-x2*y1)/(y2-y1)

        x1, y1, x2, y2 = border2
        self.border2_a = (y2-y1)/(x2-x1)
        self.border2_b = (y1*x2-y2*x1)/(x2-x1)

        self.first_detected_before_border1 = False
        self.detected_after_border2 = False

    def update(self, centroid):
        x, y = centroid

        # border 1 check
        x1, y1, x2, y2 = self.border1
        x_border = y * self.border1_a + self.border1_b
        if y1 < y < y2:
            if x < x_border:                    # to the left from the border 1
                if not self.detected_after_border2:
                    self.first_detected_before_border1 = True

        # border 2 check
        x1, y1, x2, y2 = self.border2
        y_border = x * self.border2_a + self.border2_b
        if x1 < x < x2:
            if y_border < y:                    # below the border 2
                self.detected_after_border2 = True

    @property
    def passed(self):
        return self.first_detected_before_border1 and self.detected_after_border2


class ObjectData:
    def __init__(self, object_id, centroid):
        self.object_id = object_id
        self.trajectory = [centroid]
        self.color = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
        self.north_control = NorthBoundController(border=(810, 460, 1070, 460), margin=5)
        self.turn1_control = RightTurnController(border1=(530, 650, 530, 1080), border2=(530, 930, 1470, 770))

    def update_position(self, centroid):
        self.trajectory.append(centroid)

        self.north_control.update(centroid)
        if self.north_control.passed:
            self.color = (255, 255, 255)

        self.turn1_control.update(centroid)
        if self.turn1_control.passed:
            self.color = (255, 255, 255)


class TrajectoryTracker:
    def __init__(self):
        self.objects = {}

    def update(self, current_objects_centroids):
        processed_ids = set()

        for (object_id, centroid) in current_objects_centroids.items():
            processed_ids.add(object_id)

            if object_id in self.objects:
                self.objects[object_id].update_position(centroid)
            else:
                self.objects[object_id] = ObjectData(object_id, centroid)

        removed_ids = set(self.objects.keys()).difference(processed_ids)
        for object_id in removed_ids:
            del self.objects[object_id]


def plot_trajectory(img, color, points):
    p1 = None
    for p2 in points:
        if p1 is None:
            p1 = p2
            continue
        cv2.line(img, tuple(p1), tuple(p2), color, 2)
        p1 = p2


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

    # label = f'{CLASSES[cls]}'
    # tf = max(tl - 1, 1)  # font thickness
    # t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    #
    # corner1 = c1[0], c2[1] - t_size[1] - 3
    # corner2 = c1[0] + t_size[0], c2[1]
    #
    # cv2.rectangle(img, corner1, corner2, color, -1, cv2.LINE_AA)  # filled
    # cv2.putText(img, label, (corner1[0], corner2[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
    #             lineType=cv2.LINE_AA)


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
        trajectory_trackers = {}
        north_bound_passed = {}
        right_turn_passed = {}
        for cl in tracked_classes:
            c_trackers[cl] = CentroidTracker(max_disappeared=5, max_distance=50)
            trajectory_trackers[cl] = TrajectoryTracker()
            north_bound_passed[cl] = set()
            right_turn_passed[cl] = set()

        total_frames = 0

        while True:
            frame = vs.read()
            frame = frame[1]

            if frame is None:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detections = model.detect(rgb)

            # north border
            cv2.line(frame, (810, 460), (1070, 460), (0, 192, 0), 2)

            # right turn borders
            cv2.line(frame, (530, 650), (530, 1080), (0, 192, 0), 2)
            cv2.line(frame, (530, 930), (1470, 770), (0, 192, 0), 2)

            rects = {}
            for cl in tracked_classes:
                rects[cl] = []

            for i in range(detections.shape[0]):
                cl = int(detections[i, 5])
                rects[cl].append(tuple(detections[i, :4]))
                plot_one_box(detections[i, :4], frame, cls=cl)

            for cl in tracked_classes:
                objects = c_trackers[cl].update(rects[cl])

                trajectory_trackers[cl].update(objects)

                for (object_id, object_data) in trajectory_trackers[cl].objects.items():
                    if object_data.north_control.passed:
                        north_bound_passed[cl].add(object_id)
                    if object_data.turn1_control.passed:
                        right_turn_passed[cl].add(object_id)

                    text = f'{CLASSES[cl]} {object_id}'
                    plot_trajectory(frame, object_data.color, object_data.trajectory)
                    cv2.putText(frame, text, (object_data.trajectory[-1][0] - 10, object_data.trajectory[-1][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, object_data.color, 2)

            writer.write(frame)

            total_frames += 1

            if total_frames % 30 == 0:
                print('frame {} of {} '.format(total_frames, vs_length))

        if total_frames % 30 != 0:
            print('frame {} of {} '.format(total_frames, vs_length))

        s_det = ''
        for cl in tracked_classes:
            s_det += f'{CLASSES[cl]}: north {len(north_bound_passed[cl])} near right turn {len(right_turn_passed[cl])}; '
        print(s_det)

        vs.release()
        writer.release()

        with open(counts_path, 'a') as f:
            f.write(f'{os.path.split(video_path_in)[1]} - {s_det}\n')
