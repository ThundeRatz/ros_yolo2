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

#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>
#include <thread>

#include "darknet/yolo2.h"

namespace
{
darknet::Detector yolo;
ros::Publisher publisher;
image im = {};
float *image_data = nullptr;
ros::Time timestamp;
std::mutex mutex;
std::condition_variable im_condition;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  im = yolo.convert_image(msg);
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (image_data)
      // Dropped image (image callback is running faster than YOLO thread)
      free(image_data);
    timestamp = msg->header.stamp;
    image_data = im.data;
  }
  im_condition.notify_one();
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

    image_transport::ImageTransport transport = image_transport::ImageTransport(node);
    subscriber = transport.subscribe("image", 1, imageCallback);
    publisher = node.advertise<yolo2::ImageDetections>("detections", 5);

    yolo_thread = new std::thread(run_yolo);
  }

  ~Yolo2Nodelet()
  {
    yolo_thread->join();
    delete yolo_thread;
  }

 private:
  image_transport::Subscriber subscriber;
  std::thread *yolo_thread;

  static void run_yolo()
  {
    while (ros::ok())
    {
      float *data;
      ros::Time stamp;
      {
        std::unique_lock<std::mutex> lock(mutex);
        im_condition.wait(lock, []{return image_data;});
        data = image_data;
        image_data = nullptr;
        stamp = timestamp;
      }
      boost::shared_ptr<yolo2::ImageDetections> detections(new yolo2::ImageDetections);
      *detections = yolo.detect(data);
      detections->header.stamp = stamp;
      publisher.publish(detections);
      free(data);
    }
  }
};
}  // namespace yolo2

PLUGINLIB_EXPORT_CLASS(yolo2::Yolo2Nodelet, nodelet::Nodelet)
