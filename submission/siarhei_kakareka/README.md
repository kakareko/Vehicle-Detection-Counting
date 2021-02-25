# Environment creation and activation

```bash
conda env create -f environment.yml
conda activate vehicledc_ska
```

# Topic 1

Execution

```bash
cd topic1/yolov5
python detect.py --source ../../../../datasets/predict/ --weights yolov5s.pt --save-counts --classes 2 5 7 --conf 0.25
```

This script reads images from `datasets/predict` folder of the project, and
performs detection of cars, buses and trucks.

The output data is saved into folder `topic1/yolov5/runs/detect/exp<N>`.
The list of images and number of cars, trucks and buses detected there are saved in `counts.txt`.
The images in this folder have bounding boxes around the detected objects.

# Topic 2

Execution

```bash
cd topic2/yolov5
python counter.py
```

This script reads videos from the `videos` folder of the project, and
performs detection of cars, buses and trucks.

The output data is saved into folder `topic2/yolov5/out/`.
The list of videos and the number of cars, trucks and buses detected there are saved in `counts.txt`.
The videos in this folder show bounding boxes around the detected objects, and each detected 
object have an ID that is shown in the centre of the box.

The approach computes distances between centers of detected objects on two consecutive frames. 
The objects with minimal distance between them on the different frames are treated as same object.
If the distance is higher than some threshold value, the objects are treated as separate.

Note:
The high number of counted vehicles is mainly due to the loss of the detections in
case of occlusions. Therefore, the vehicles, that were detected, then lost for a
number of frames and then detected again, will be counted as separate vehicles.
The objects detector and the tracking algorithm have to be improved to increase the
counting accuracy.

# Topic 3

Execution

```bash
cd topic3/yolov5
python counter.py
```

This script tracks trajectories of vehicles and counts cars, buses and trucks that
go to the north and turn right on the near end of the crossroad in the video
`video/Topic23-video1.mp4`.

The output data is saved into folder `topic3/yolov5/out/`.
The name of the video file and the counted figures are saved to `counts.txt`.