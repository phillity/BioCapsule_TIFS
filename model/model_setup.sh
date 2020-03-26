#!/bin/bash

cd model

echo "Downloading and extracting face models!"
wget https://cs.iupui.edu/~phillity/model.zip
unzip model.zip
mv det?* mtcnn/
mv 20180408-102900.pb facenet/20180408-102900.pb
rm model.zip
