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

echo "Downloading and extracting LFW images!"
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
wget http://vis-www.cs.umass.edu/lfw/pairs.txt
wget http://vis-www.cs.umass.edu/lfw/people.txt
tar -xvzf lfw.tgz
rm lfw.tgz

echo "Downloading and extracting GTDB images!"
wget http://www.anefian.com/research/gt_db.zip
unzip gt_db.zip
rm gt_db.zip
mv gt_db gtdb
cd gtdb
find . -name "*.jbf" -type f -delete
cd ..

echo "Downloading and extracting Caltech Faces images!"
mkdir caltech
cd caltech
wget http://www.vision.caltech.edu/Image_Datasets/faces/faces.tar
tar -xvf faces.tar
mkdir s01
for i in {0001..0021}; do mv image_$i.jpg s01/image_$i.jpg; done
mkdir s02
for i in {0022..0041}; do mv image_$i.jpg s02/image_$i.jpg; done
mkdir s03
for i in {0042..0046}; do mv image_$i.jpg s03/image_$i.jpg; done
mkdir s04
for i in {0047..0068}; do mv image_$i.jpg s04/image_$i.jpg; done
mkdir s05
for i in {0069..0089}; do mv image_$i.jpg s05/image_$i.jpg; done
mkdir s06
for i in {0090..0112}; do mv image_$i.jpg s06/image_$i.jpg; done
mkdir s07
for i in {0113..0132}; do mv image_$i.jpg s07/image_$i.jpg; done
mkdir s08
for i in {0133..0137}; do mv image_$i.jpg s08/image_$i.jpg; done
mkdir s09
for i in {0138..0158}; do mv image_$i.jpg s09/image_$i.jpg; done
mkdir s10
for i in {0159..0165}; do mv image_$i.jpg s10/image_$i.jpg; done
mkdir s11
for i in {0166..0170}; do mv image_$i.jpg s11/image_$i.jpg; done
mkdir s12
for i in {0171..0175}; do mv image_$i.jpg s12/image_$i.jpg; done
mkdir s13
for i in {0176..0195}; do mv image_$i.jpg s13/image_$i.jpg; done
mkdir s14
for i in {0196..0216}; do mv image_$i.jpg s14/image_$i.jpg; done
mkdir s15
for i in {0217..0241}; do mv image_$i.jpg s15/image_$i.jpg; done
mkdir s16
for i in {0242..0263}; do mv image_$i.jpg s16/image_$i.jpg; done
mkdir s17
for i in {0264..0268}; do mv image_$i.jpg s17/image_$i.jpg; done
mkdir s18
for i in {0269..0287}; do mv image_$i.jpg s18/image_$i.jpg; done
mkdir s19
for i in {0288..0307}; do mv image_$i.jpg s19/image_$i.jpg; done
mkdir s20
for i in {0308..0336}; do mv image_$i.jpg s20/image_$i.jpg; done
mkdir s21
for i in {0337..0356}; do mv image_$i.jpg s21/image_$i.jpg; done
mkdir s22
for i in {0357..0376}; do mv image_$i.jpg s22/image_$i.jpg; done
mkdir s23
for i in {0377..0398}; do mv image_$i.jpg s23/image_$i.jpg; done
mkdir s24
for i in {0404..0408}; do mv image_$i.jpg s24/image_$i.jpg; done
mkdir s25
for i in {0409..0428}; do mv image_$i.jpg s25/image_$i.jpg; done
mkdir s26
for i in {0429..0450}; do mv image_$i.jpg s26/image_$i.jpg; done
rm faces.tar ImageData.mat README image_0399.jpg image_0400.jpg image_0401.jpg image_0402.jpg image_0403.jpg
cd ..

echo "Downloading and extracting RS images!"
wget https://cs.iupui.edu/~phillity/rs.tar.gz
tar -xvzf rs.tar.gz
rm rs.tar.gz
