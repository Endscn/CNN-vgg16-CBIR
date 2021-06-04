# CNN-vgg16-CBIR
use CNN CBIR and make CNN model

2021-04-13 19:10
 idea launching
 
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

# 4-1. Find Outlier and Remove

1) Outlier detection
2) Mean or Median or what

## Version
    
    Python 3.6.5
    tensorflow==1.13.1
    tensorflow-estimator==1.13.0
    tensorflow-gpu==1.13.1
    opencv-python==4.4.0.46
    Keras==2.3.1
    imutils==0.5.4
    h5py==2.10.0
    protobuf==3.14.0

## log

    2021-04-26 19:19
    labeling done

    2021-04-28 18:00
    use InceptionV3 model
    Accuracy not converge, Change opt RMSprop -> Adam  & Corrected the learning rate.

    2021-05-04 18:49
    data cleaning sooo haaaard
    14490 -> 8769   img label appropriately.

    2021-05-07 19:42
    acc == 0.68
    rgba2rgb issue

    2021-05-14 13:45
    final cnn test update 
    test acc= 0.65
    little lack of classification criteria 
    
    2021-05-16 21:18
    outlier detection has begin
    
    2021-05-28
    representative image setting
    
    2021-06-04 19:37
    
    model completion
    random_picture_acc = 0.75
