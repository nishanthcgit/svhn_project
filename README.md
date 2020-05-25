# PICQUORA
Programming task for Detection digits in the svhn dataset

Credit to https://github.com/pavitrakumar78 for construct_dataseta.py

Credit to https://github.com/zzh8829/yolov3-tf2 for yolov3 model

1)Use construct_data and set the folder paths to process your data into dataframes and convert them into tfrecord format

2)Train yolov3 using the instructions given at https://github.com/zzh8829/yolov3-tf2/blob/master/docs/training_voc.md - here I faced issue with an ongoing Tensorflow bug - might want to turn off validation here for now. 

3) Move detect_many into the yolov3-tf2 folder, set your input and out put folder paths and run it to detect  on your own list of images
