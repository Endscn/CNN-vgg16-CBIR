# Tree

├── database/

├── resultcsv/

├── extract_cnn_vgg16_keras.py

├── query_online.py

└── index.py



1. database 폴더, resultcsv 폴더 만든다.   > Make "database" folder , "resultcsv" folder

2. database에 모든 이미지 다 넣는다. 이미지 이름은 번호로 하는게 좋다. ex) 1.jpg, 2.jpg...  > Put all imgs in the "database" folder 

3. python index.py -database database -index featureCNN.h5 
터미널에서 실행하여 featureCNN.h5 파일 만들어준다.    > Enter this code in terminal, "featureCNN.h5" file has been created.

4. python query_online.py -query database/1135.jpg -index featureCNN.h5 -result database
터미널에서 실행하여 CSV 파일 만들어준다.  > Enter this code in terminal, "~.csv" file has been created.
  
5. 이후 CSV 파일을 통합해서 분석   > Generated file can be useful for analysis.
