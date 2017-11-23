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

#include "darknet/yolo2.h"

#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <yolo2/Detection.h>
#include <yolo2/ImageDetections.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

extern "C"
{
#undef __cplusplus
#include "detection_layer.h"  // NOLINT(build/include)
#include "parser.h"  // NOLINT(build/include)
#include "region_layer.h"  // NOLINT(build/include)
#include "utils.h"  // NOLINT(build/include)
#define __cplusplus
}

namespace darknet
{
void Detector::load(std::string& model_file, std::string& trained_file, double min_confidence, double nms)
{
  min_confidence_ = min_confidence;
  nms_ = nms;
  net_ = parse_network_cfg(&model_file[0]);
  load_weights(net_, &trained_file[0]);
  set_batch_network(net_, 1);

  layer output_layer = net_->layers[net_->n - 1];
  boxes_.resize(output_layer.w * output_layer.h * output_layer.n);
  probs_.resize(output_layer.w * output_layer.h * output_layer.n);
  float *probs_mem = static_cast<float *>(calloc(probs_.size() * output_layer.classes, sizeof(float)));
  for (auto& i : probs_)
  {
    i = probs_mem;
    probs_mem += output_layer.classes;
  }
}

Detector::~Detector()
{
  free(probs_[0]);
  free_network(net_);
}

yolo2::ImageDetections Detector::detect(float *data, int original_width, int original_height)
{
  yolo2::ImageDetections detections;
  detections.detections = forward(data, original_width, original_height);
  return detections;
}

image Detector::convert_image(const sensor_msgs::ImageConstPtr& msg)
{
  if (msg->encoding != sensor_msgs::image_encodings::RGB8)
  {
    ROS_ERROR("Unsupported encoding");
    exit(-1);
  }

  auto data = msg->data;
  uint32_t height = msg->height, width = msg->width, offset = msg->step - 3 * width;
  uint32_t i = 0, j = 0;
  image im = make_image(width, height, 3);

  for (uint32_t line = height; line; line--)
  {
    for (uint32_t column = width; column; column--)
    {
      for (uint32_t channel = 0; channel < 3; channel++)
        im.data[i + width * height * channel] = data[j++] / 255.;
      i++;
    }
    j += offset;
  }

  if (net_->w == (int)width && net_->h == (int)height)
  {
    return im;
  }
  image resized = letterbox_image(im, net_->w, net_->h);
  free_image(im);
  return resized;
}

std::vector<yolo2::Detection> Detector::forward(float *data, int original_width, int original_height)
{
  float *prediction = network_predict(net_, data);
  layer output_layer = net_->layers[net_->n - 1];

  output_layer.output = prediction;
  if (output_layer.type == DETECTION)
    get_detection_boxes(output_layer, 1, 1, min_confidence_, probs_.data(), boxes_.data(), 0);
  else if (output_layer.type == REGION)
    get_region_boxes(output_layer, original_width, original_height, net_->w, net_->h, min_confidence_, probs_.data(), boxes_.data(), 0, 0, 0, .5, 1);
  else
    error("Last layer must produce detections\n");

  int num_classes = output_layer.classes;
  do_nms(boxes_.data(), probs_.data(), output_layer.w * output_layer.h * output_layer.n, num_classes, nms_);
  std::vector<yolo2::Detection> detections;
  for (unsigned i = 0; i < probs_.size(); i++)
  {
    int class_id = max_index(probs_[i], num_classes);
    float prob = probs_[i][class_id];
    if (prob)
    {
      yolo2::Detection detection;
      box b = boxes_[i];

      detection.x = b.x;
      detection.y = b.y;
      detection.width = b.w;
      detection.height = b.h;
      detection.confidence = prob;
      detection.class_id = class_id;
      detections.push_back(detection);
    }
  }
  return detections;
}
}  // namespace darknet
