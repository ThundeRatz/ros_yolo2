# Scripts

## labelImg_to_yolo.py

Converts [labelImg](https://github.com/tzutalin/labelImg) formatted labels (VOC format) to YOLO formatted labels. Similar to Darknet's script, but walks a source directory looking for labels instead of assuming the directory format used in VOCdevkit and has a few more features (generating validation/training splits, class names file and data file).

20% of the labels are taken randomly for validation and 80% for training. YOLO requires that the labels are placed in a folder called `labels` and the images in a folder called `images`, and these folders must be contained in the same folder.

##### Example usage:

`./labelImg_to_yolo.py directory_with_voc_labels directory_with_images_folder`

Given a directory containing the `images` folder, the script will create and populate the `labels` folder, create a `backup` folder, create files `training-list.txt`, `validation-list.txt` and `names.txt` and create `dataset.data` pointing to these files.

YOLO training can be run with:

`darknet detector train path_to_dataset.data network_config initial_weights`

`network_config` is the network .cfg file (eg. `yolo.cfg`) and `initial_weights` are optional weights to initialize the network, which can be obtained at [Darknet's site](http://pjreddie.com/darknet/imagenet/#darknet19_448). Use of `darknet19_448.conv.23` is recommended in YOLO's page.
