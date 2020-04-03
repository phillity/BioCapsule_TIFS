#!/bin/bash

cd data

echo "Downloading and extracting FaceNet features!"
wget https://cs.iupui.edu/~phillity/facenet_tifs.tar.gz
tar -xvf facenet_tifs.tar.gz

echo "Downloading and extracting ArcFace features!"
wget https://cs.iupui.edu/~phillity/arcface_tifs.tar.gz
tar -xvf arcface_tifs.tar.gz
