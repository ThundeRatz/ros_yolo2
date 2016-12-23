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

#include <sensor_msgs/image_encodings.h>

#include <cstdint>
#include <cstdlib>

extern "C"
{
#undef __cplusplus
#include "detection_layer.h"
#include "parser.h"
#include "region_layer.h"
#include "utils.h"
#define __cplusplus
}

namespace darknet
{
Detector::Detector(std::string& model_file, std::string& trained_file) : predictions(FRAMES), prediction_index(0), filled_buffer(false)
{
  net = parse_network_cfg(&model_file[0]);
  load_weights(&net, &trained_file[0]);
  set_batch_network(&net, 1);

  layer output_layer = net.layers[net.n - 1];
  average.resize(output_layer.outputs);
  boxes.resize(output_layer.w * output_layer.h * output_layer.n);
  probs.resize(output_layer.w * output_layer.h * output_layer.n);
  for (auto& i : probs)
    i = (float *)calloc(output_layer.classes, sizeof(float));
  for (auto& i : predictions)
    i = (float *)calloc(output_layer.outputs, sizeof(float));
}

Detector::~Detector()
{
  for (auto& i : probs)
    free(i);
  free_network(net);
}

yolo2::ImageDetections Detector::detect(const sensor_msgs::ImageConstPtr& msg)
{
  if (msg->encoding != sensor_msgs::image_encodings::RGB8)
  {
    ROS_ERROR("Ignoring unsupported encoding ", msg->encoding);
    return yolo2::ImageDetections();
  }
  image im = convert_image(msg);
  yolo2::ImageDetections detections;
  detections.detections = forward(im.data);
  detections.image_stamp = msg->header.stamp;
  free_image(im);
  return detections;
}

image Detector::convert_image(const sensor_msgs::ImageConstPtr& msg)
{
  auto data = msg->data;
  uint32_t height = msg->height, width = msg->width, offset = msg->step - 3 * width;
  uint32_t i = 0, j = 0;
  image im = make_image(width, height, 3);

  for (uint32_t line = height; line; line--)
  {
    for (uint32_t column = width; column; column--) {
      for (uint32_t channel = 0; channel < 3; channel++)
        im.data[i + width * height * channel] = data[j++] / 255.;
      i++;
    }
    j += offset;
  }

  image resized = resize_image(im, net.w, net.h);
  free_image(im);
  return resized;
}

std::vector<yolo2::Detection> Detector::forward(float *data)
{
  float *prediction = network_predict(net, data);
  layer output_layer = net.layers[net.n - 1];

  memcpy(predictions[prediction_index], prediction, output_layer.outputs * sizeof(float));
  if (++prediction_index == FRAMES)
  {
    filled_buffer = true;
    prediction_index = 0;
  }
  if (!filled_buffer)
    return std::vector<yolo2::Detection>();

  mean_arrays(predictions.data(), FRAMES, output_layer.outputs, average.data());
  output_layer.output = average.data();

  if (output_layer.type == DETECTION)
    get_detection_boxes(output_layer, 1, 1, .8, probs.data(), boxes.data(), 0);
  else if (output_layer.type == REGION)
    get_region_boxes(output_layer, 1, 1, .8, probs.data(), boxes.data(), 0, 0);
  else
    error("Last layer must produce detections\n");

  int num_classes = output_layer.classes;
  do_nms(boxes.data(), probs.data(), output_layer.w * output_layer.h * output_layer.n, num_classes, .4);
  std::vector<yolo2::Detection> detections;
  for (unsigned i = 0; i < probs.size(); i++)
  {
    int class_id = max_index(probs[i], num_classes);
    float prob = probs[i][class_id];
    if (prob > .8)
    {
      yolo2::Detection detection;
      box b = boxes[i];

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
