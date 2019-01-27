#!/bin/sh


while read p; do
  gsutil cp gs://uga-dsp/project1/data/bytes/$p.bytes ../dataset/data/test/
done < ../dataset/files/X_small_test.txt

while read p; do
  gsutil cp gs://uga-dsp/project1/data/bytes/$p.bytes ../dataset/data/train/
done < ../dataset/files/X_small_train.txt
