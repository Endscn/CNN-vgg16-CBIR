# CNN-vgg16-CBIR
use CNN CBIR and make CNN model

2021-04-13 19:10

2021-04-19 17:15

## IDEAS -- How can we classify "Emotional(not formed)" images?

# 1. Make image same size

1) make img size 1500x1500 
2) jpg2png , png2jpg

# 2. IMG sort (USE CNN CBIR and LABELING)

1) classify img by vgg16 model
2) analysis csv file 
3) label img with highest similarity
4) Gather several group (similar label) (big calligraphy, small calligraphy)

# 3. Fit CNN model

1) use labeled img
2) our model
2-1) transfer learning model
3) compare each other
4) use "model.predict" and save result in database

# 4. Find Similar IMG

1) sort with result 
2) show similar img


## Version

    tensorflow==1.13.1
    tensorflow-estimator==1.13.0
    tensorflow-gpu==1.13.1
    opencv-python==4.4.0.46
    Keras==2.3.1
    imutils==0.5.4
    h5py==2.10.0
    protobuf==3.14.0





