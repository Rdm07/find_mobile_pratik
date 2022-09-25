import os, sys
import random
import torch
import cv2
import numpy as np
import PIL.Image as Image
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

def load_imgs_and_labels(training_data_path, val_ratio):
	img_trains = []
	img_vals = []
	all_img = []
	all_lab = []

	lab_file = open(os.path.join(training_data_path, 'labels.txt'))
	lines = lab_file.readlines()
	labels = {}
	for line in lines:
		spl = line.split()
		labels[spl[0]] = [float(spl[1]), float(spl[2])]

	for file in os.listdir(training_data_path):
		if ".jpg" in file:
			all_img.append((os.path.join(training_data_path, file), labels[file][0], labels[file][1]))
	valid_ind = random.sample(list(range(len(all_img))), int(0.2*len(all_img)))

	for i in range(len(all_img)):
		if i in valid_ind:
			img_vals.append(all_img[i])
		else:
			img_trains.append(all_img[i])

	return img_trains, img_vals

class data_loader(Dataset):
	"""
	Dataset to read image and label for training
	"""
	def __init__(self, imgs, transform=None):
		self.imgs = imgs
		self.transform = transform
	def __getitem__(self, index):
		img = self.imgs[index]
		png = Image.open(img[0]).convert('RGB')			# ori: RGB, do not convert to numpy, keep it as PIL image to apply transform
		lab_dict = torch.tensor([img[1], img[2]])
		
		if self.transform:
			png = self.transform(png)

		return png, lab_dict

	def __len__(self):
		return len(self.imgs)

def parallelize_model(model):
	if torch.cuda.is_available():
		model = model.cuda()
		model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
		cudnn.benchmark = True
	return model

def unparallelize_model(model):
	try:
		while 1:
			# to avoid nested dataparallel problem
			model = model.module
	except AttributeError:
		pass
	return model