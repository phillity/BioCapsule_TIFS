#!/bin/bash

cd src/model

echo "Downloading and extracting ArcFace model!"
wget https://cs.iupui.edu/~phillity/arcface.tar.gz
tar -xvzf arcface.tar.gz
rm arcface.tar.gz

echo "Downloading and extracting FaceNet model!"
wget https://cs.iupui.edu/~phillity/facenet.tar.gz
tar -xvzf facenet.tar.gz
rm facenet.tar.gz

echo "Downloading and extracting MTCNN model!"

wget https://cs.iupui.edu/~phillity/mtcnn.tar.gz
tar -xvzf mtcnn.tar.gz
rm mtcnn.tar.gz
