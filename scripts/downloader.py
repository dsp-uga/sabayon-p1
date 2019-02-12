#! /usr/bin/python3

import os

def X_small_train():
	with open('../dataset/files/X_small_train.txt') as fp:
		for filename in fp.read().split('\n'):
			command = 'gsutil cp gs://uga-dsp/project1/data/bytes/' + filename + '.bytes ../dataset/data/train'
			os.system(command)

def X_small_test():
	with open('../dataset/files/X_small_test.txt') as fp:
		for filename in fp.read().split('\n'):
			command = 'gsutil cp gs://uga-dsp/project1/data/bytes/' + filename + '.bytes ../dataset/data/test'
			os.system(command)

def download_one():
	filename='JkcvVWjUdD0OTuSmzA4q'
	command = 'gsutil cp gs://uga-dsp/project1/data/bytes/' + filename + '.bytes ../dataset/data/train' 
	os.system(command)


download_one()
