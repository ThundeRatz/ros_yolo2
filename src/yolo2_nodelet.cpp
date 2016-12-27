/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 ThundeRatz

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/console.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <yolo2/ImageDetections.h>

#include <string>
#include <vector>

#include "darknet/yolo2.h"

namespace
{
  darknet::Detector yolo;
  ros::Publisher publisher;

  void imageCallback(const sensor_msgs::ImageConstPtr& msg)
  {
    publisher.publish(yolo.detect(msg));
  }
}  // namespace

namespace yolo2
{
class Yolo2Nodelet : public nodelet::Nodelet
{
 public:
  virtual void onInit()
  {
    ros::NodeHandle& node = getPrivateNodeHandle();
    const std::string NET_DATA = ros::package::getPath("yolo2") + "/data/";
    std::string config = NET_DATA + "yolo.cfg", weights = NET_DATA + "yolo.weights";
    double confidence, nms;
    node.param<double>("confidence", confidence, .8);
    node.param<double>("nms", nms, .4);
    yolo.load(config, weights, confidence, nms);

    transport = new image_transport::ImageTransport(node);
    subscriber = transport->subscribe("image", 1, imageCallback);
    publisher = node.advertise<yolo2::ImageDetections>("detections", 5);
  }

  ~Yolo2Nodelet()
  {
    delete transport;
  }

 private:
  image_transport::ImageTransport *transport;
  image_transport::Subscriber subscriber;
};
}  // namespace yolo2

PLUGINLIB_EXPORT_CLASS(yolo2::Yolo2Nodelet, nodelet::Nodelet)
