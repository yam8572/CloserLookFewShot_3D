#!/usr/bin/env bash
##wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
##wget -O file https://googledrive.com/host/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45
##wget 'https://docs.google.com/uc?export=download&id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45' -O CUB_200_2011.tgz 
##wget "https://drive.google.com/uc?export=download&id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45"
##wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45" -O CUB_200_2011.tgz
fileId=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45
fileName=CUB_200_2011.tgz
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}

tar -zxvf CUB_200_2011.tgz
python write_CUB_filelist.py
