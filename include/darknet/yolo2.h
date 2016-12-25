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

#ifndef INCLUDE_DARKNET_YOLO2_H_
#define INCLUDE_DARKNET_YOLO2_H_

#include <image_transport/image_transport.h>
#include <yolo2/Detection.h>
#include <yolo2/ImageDetections.h>

#include <string>
#include <vector>

extern "C"
{
#undef __cplusplus
#include "box.h"
#include "image.h"
#include "network.h"
#define __cplusplus
}

namespace darknet
{
class Detector
{
 public:
  Detector(std::string& model_file, std::string& trained_file, double min_confidence, double nms);
  ~Detector();
  yolo2::ImageDetections detect(const sensor_msgs::ImageConstPtr& msg);

 private:
  image convert_image(const sensor_msgs::ImageConstPtr& msg);
  std::vector<yolo2::Detection> forward(float *data);

  double min_confidence, nms;
  network net;
  std::vector<box> boxes;
  std::vector<float *> probs;
};
}  // namespace darknet

#endif  // INCLUDE_DARKNET_YOLO2_H_
