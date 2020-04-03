#!/bin/bash

vggface2_username=$1
vggface2_password=$2

cd image

echo "Downloading and extracting VGGFace2 train images!"
python vggface2_download.py train $vggface2_username $vggface2_password
tar -xvzf vggface2_train.tar.gz
rm vggface2_train.tar.gz
echo "Downloading and extracting VGGFace2 test images!"
python vggface2_download.py test $vggface2_username $vggface2_password
tar -xvzf vggface2_test.tar.gz
rm vggface2_test.tar.gz
mkdir vggface2
mv -v train/* vggface2/
mv -v test/* vggface2/
rm -r train test
