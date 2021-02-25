import torch
import numpy as np

from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from models.experimental import attempt_load


class YOLOv5ObjectDetector(object):
    def __init__(self, weights='weights/yolov5s.pt', imgsz=384, conf_thres=0.25, iou_thres=0.45, classes=None):
        """

        :param weights: model.pt path(s)
        :param imgsz: inference size (pixels)
        :param conf_thres: object confidence threshold
        :param iou_thres: IOU threshold for NMS
        :param classes: filter by class
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes

        # Initialize
        self.device = select_device()
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        names = self.model.names
        if classes is None:
            self.names = names
        else:
            self.names = [names[i] if i in classes else None for i in range(len(names))]

        # init inference
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once

    @property
    def class_names(self):
        return self.names

    def detect(self, img):
        """
        Detect objects on image
        :param img:
            numpy array with RGB image of shape HxWx3
        :return:
            numpy array with detections with shape: nx6 (x1, y1, x2, y2, confidence, class)
        """

        input_img_shape = img.shape

        # Padded resize
        img = letterbox(img, new_shape=self.imgsz)[0]

        # Convert
        img = img.transpose(2, 0, 1)  # to 3x416x416
        img = np.ascontiguousarray(img)  # slight performance improvement

        # Normalize
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes)[0]

        # Scale back
        pred[:, :4] = scale_coords(img.shape[-2:], pred[:, :4], input_img_shape).round()

        pred = pred.cpu().numpy()

        return pred
