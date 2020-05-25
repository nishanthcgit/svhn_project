# SVHN DETECTION
Programming task for Detection digits in the svhn dataset

Credit to https://github.com/pavitrakumar78 for construct_data.py

Credit to https://github.com/zzh8829/yolov3-tf2 for yolov3 model

1)Use construct_data and set the folder paths to process your data into dataframes and convert them into tfrecord format

2)Train yolov3 using the instructions given at https://github.com/zzh8829/yolov3-tf2/blob/master/docs/training_voc.md - here I faced an issue with an ongoing Tensorflow bug - might want to turn off validation here for now. (use Transfer learning, random weight training is very slow)

3)Move detect_many into the yolov3-tf2 folder, set your input and output folder paths and run it to detect  on your own list of images. 

Additional steps:

-Given more time, there are a lot of steps that can be taken to improve accuracy - data augmentation to start with, using the extra images to increase the size of the training set, hyperparameter tuning, anchor optimization and so on..

-Make code clean and usable in other situations - this is very hastily put together.

Using yolo might be overkill for this task but its a fast model that has many available implementations - I had less than a day to put this together so I used pre written models mainly.

