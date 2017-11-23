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

#ifndef DARKNET_YOLO2_H
#define DARKNET_YOLO2_H

#include <image_transport/image_transport.h>
#include <yolo2/Detection.h>
#include <yolo2/ImageDetections.h>

#include <string>
#include <vector>

extern "C"
{
#undef __cplusplus
#include "box.h"  // NOLINT(build/include)
#include "image.h"  // NOLINT(build/include)
#include "network.h"  // NOLINT(build/include)
#define __cplusplus
}

namespace darknet
{
class Detector
{
 public:
  Detector() {}
  void load(std::string& model_file, std::string& trained_file, double min_confidence, double nms);
  ~Detector();
  image convert_image(const sensor_msgs::ImageConstPtr& msg);
  yolo2::ImageDetections detect(float *data, int original_width, int original_height);

 private:
  std::vector<yolo2::Detection> forward(float *data, int original_width, int original_height);

  double min_confidence_, nms_;
  network *net_;
  std::vector<box> boxes_;
  std::vector<float *> probs_;
};
}  // namespace darknet

#endif  // DARKNET_YOLO2_H
