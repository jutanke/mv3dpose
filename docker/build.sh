#!/usr/bin/env bash

cp Dockerfile Dockerfile.bkp
#NEW_UUID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
#echo "# $NEW_UUID" >> Dockerfile
echo "RUN adduser --disabled-password --gecos \"\" -u $UID user"  >> Dockerfile
echo "USER user" >> Dockerfile

docker build --tag='jutanke/mv3dpose' .


rm Dockerfile
mv Dockerfile.bkp Dockerfile