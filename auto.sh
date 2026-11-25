#!/bin/bash

cd /var/www/html/Website_Volcanic

mkdir -p data/terrain/Copernicus_GLO30
wget -O data/terrain/Copernicus_GLO30/tileList.txt https://copernicus-dem-30m.s3.amazonaws.com/tileList.txt

ls -lh data/terrain/Copernicus_GLO30/
wc -l data/terrain/Copernicus_GLO30/tileList.txt

mkdir -p data/terrain/natural_earth

wget https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip -O /tmp/ne_10m_land.zip

unzip -o /tmp/ne_10m_land.zip -d data/terrain/natural_earth/

ls -lh data/terrain/natural_earth/
