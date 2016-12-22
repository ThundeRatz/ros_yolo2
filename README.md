# ROS YOLOv2 integration
Integrates [YOLOv2](http://pjreddie.com/darknet/yolo/) with ROS, generating messages with position and confidence of detection.

## Requirements
* ROS (tested on kinetic)

* GPU supporting CUDA

## Installation
This repository should be cloned to the `src` directory of your workspace. Alternatively, the following `.rosinstall` entry adds this repository for usage with `wstool`:

```
- git:
    local-name: yolo2
    uri: https://github.com/ThundeRatz/ros_yolo2.git
```

## Setup
Copy your network config to `data/yolo.cfg` and network weights as `data/yolo.weights`.

## Contributors
Written by the ThundeRatz robotics team.
