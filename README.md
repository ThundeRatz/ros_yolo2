# ROS YOLOv2 integration
Integrates [YOLOv2](http://pjreddie.com/darknet/yolo/) with ROS, generating messages with position and confidence of detection.

## Requirements
* ROS (tested on Jade and Kinetic)

* GPU supporting CUDA

## Installation
If using `wstool`, the following `.rosinstall` entry adds this repository to your workspace:

```
- git:
    local-name: yolo2
    uri: https://github.com/ThundeRatz/ros_yolo2.git
```

If building from source, change to the `src` directory of your workspace and use `git clone --recursive https://github.com/tiagoshibata/ros_yolo2.git` to download with submodules.

This repository is using ROS's [nodelet API](http://wiki.ros.org/nodelet), which allows loading multiple nodelets in the same process, supporting zero-copy message transfers between nodelets. No node executables are built; instead, shared libraries are. To use them, check the nodelet docs and [tutorials](http://wiki.ros.org/nodelet/Tutorials). This sample launchfile creates a nodelet manager with an uvc_camera and a yolo2 nodelet:

```
<launch>
  <group ns="vision">
    <node pkg="nodelet" type="nodelet" name="vision_nodelet" args="manager"/>
    <node pkg="nodelet" type="nodelet" name="camera" args="load uvc_camera/CameraNodelet vision_nodelet" />
    <remap from="yolo2/image" to="image_raw" />
    <node pkg="nodelet" type="nodelet" name="yolo2" args="load yolo2/Yolo2Nodelet vision_nodelet">
      <param name="confidence" value="0.5" />
      <param name="nms" value="0.4" />
    </node>
  </group>
</launch>
```

## Setup
Copy your network config to `data/yolo.cfg` and network weights as `data/yolo.weights`.

## Message format

For every image received, an [ImageDetections](msg/ImageDetections.msg) message is generated at the `yolo2_detections` topic, containing a Header and a vector of [Detection](msg/Detection.msg). The header time stamp is copied from the received image message; this way, a set of detections can be matched to a specific image. Each [Detection](msg/Detection.msg) contains the following attributes:

* `uint32 class_id` - Class number configured during training;
* `float32 confidence` - Confidence;
* `float32 x` - X position of the object's center relative to the image (between 0 = left and 1 = right);
* `float32 y` - Y position of the object's center relative to the image (between 0 = top and 1 = bottom);
* `float32 width` - Width of the object relative to the image (between 0 and 1);
* `float32 height` - Height of the object relative to the image (between 0 and 1).

## Contributors
Written by the ThundeRatz robotics team.
