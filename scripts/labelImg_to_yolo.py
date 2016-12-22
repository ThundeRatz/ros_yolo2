#!/bin/env python3
import os
import random
import sys
from xml.etree import ElementTree

CLASSES = ['cone']


def convert(size, box):
    dw = 1 / size[0]
    dh = 1 / size[1]
    x = (box[0] + box[1]) / 2
    y = (box[2] + box[3]) / 2
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


def process_label_file(voc_input, yolo_output):
    root = ElementTree.parse(voc_input).getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(yolo_output, 'w') as output:
        for obj in root.iter('object'):
            name = obj.find('name').text
            if name not in CLASSES:
                print('WARNING: Object {} not configured, ignoring (file {})'.format(name, voc_input))
                continue
            index = CLASSES.index(name)
            box = obj.find('bndbox')
            coordinates = [float(box.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')]
            converted = convert((w, h), coordinates)
            output.write('{} {}\n'.format(index, " ".join([str(x) for x in converted])))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Directory with VOC labels')
    parser.add_argument('output', help='Output directory, should contain a images/ folder')
    return parser.parse_args()


def main(args):
    print('Using classes: {}'.format(' '.join(CLASSES)))
    args = parse_args()
    input_dir = os.path.realpath(args.input)
    output_dir = os.path.realpath(args.output)

    label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(label_dir, exist_ok=True)
    backup_dir = os.path.join(output_dir, 'backup')
    os.makedirs(backup_dir, exist_ok=True)

    images_dir = os.path.join(output_dir, 'images')
    dataset = []
    for dirpath, _, filenames in os.walk(input_dir):
        for label in filter(lambda x: x.endswith('.xml'), filenames):
            full_path = os.path.join(dirpath, label)
            output = os.path.join(label_dir, label[:-3] + 'txt')
            dataset.append(os.path.join(images_dir, label[:-3] + 'png'))
            process_label_file(full_path, output)

    random.shuffle(dataset)
    validation_image_count = len(dataset) // 5
    train = os.path.join(output_dir, 'training-list.txt')
    with open(train, 'w') as f:
        f.writelines('\n'.join(dataset[validation_image_count + 1:]))
    validation = os.path.join(output_dir, 'validation-list.txt')
    with open(validation, 'w') as f:
        f.write('\n'.join(dataset[:validation_image_count]))
    names = os.path.join(output_dir, 'names.txt')
    with open(names, 'w') as f:
        f.write('\n'.join(CLASSES))
    config = {
        'classes': len(CLASSES),
        'train': train,
        'valid': validation,
        'names': names,
        'backup': backup_dir,
    }
    with open(os.path.join(output_dir, 'dataset.data'), 'w') as f:
        f.writelines('{} = {}\n'.format(k, v) for k, v in config.items())


if __name__ == '__main__':
    main(sys.argv)
