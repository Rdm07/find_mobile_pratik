from cProfile import label
import os, sys, time
import argparse
import random, copy
from xml.dom.pulldom import START_ELEMENT
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from time import strftime
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc, f1_score
from utility_functions import *

parser = argparse.ArgumentParser(description='Find Mobile Resnet18 Training')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--net_type', default='RESNET_18', type=str, help='model')
parser.add_argument('--trainer', default='adam', type=str, help='optimizer')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs in training')
parser.add_argument('--check_after', default=1, type=int, help='check the network after check_after epoch')
parser.add_argument('--data', type=str, default='none', help="path to the folder containing all subfolders of training/testing data", required=True)
parser.add_argument('--val_ratio', default=0.2, help="Set the percentage of image files set aside as validation set", required=False)
args = parser.parse_args()

rand_seed = 37
freq_print = 100							# print stats every {} batches
if rand_seed is not None:
	np.random.seed(rand_seed)
	torch.manual_seed(rand_seed)
	torch.cuda.manual_seed(rand_seed)

use_gpu = torch.cuda.is_available()
print('Using GPU: ', use_gpu)

if use_gpu == True:
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

data_folder = args.data

print("Loading Data")

data_transforms = {
	'train': transforms.Compose([
		transforms.RandomHorizontalFlip(),  # simple data augmentation
		transforms.RandomVerticalFlip(),	# simple data augmentation
		transforms.ToTensor()
		]),

	'val': transforms.Compose([
		transforms.ToTensor()
	]),
}

img_trains, img_vals = load_imgs_and_labels(training_data_path = args.data, val_ratio = args.val_ratio)

random.shuffle(img_trains)
random.shuffle(img_vals)

train_set = data_loader(img_trains, transform = data_transforms['train'])
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

val_set = data_loader(img_vals, transform = data_transforms['val'])
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

def auc_roc(Pr, Tr):
	fpr, tpr, _ = roc_curve(Tr, Pr, pos_label=1.0)
	return auc(fpr, tpr)

def distance_loss(outputs, labels):
	outputs = outputs.squeeze()
	labels = labels.squeeze()

	if outputs.size() != labels.size():
		raise Exception("Outputs and Labels Sizes do not match")
	else:
		sq_diff = torch.square(outputs - labels)
		loss = torch.sqrt(torch.sum(sq_diff))
		return loss

def val_epoch():
	pass

def train_model(model, criterion, num_epochs = 100, train_loader = train_loader, val_loader = val_loader):
	best_auc = 0
	best_epoch = 0
	start_training = time.time()

	for epoch in range(num_epochs):
		start = time.time()
		
		if epoch < 4: lr = args.lr
		elif epoch < 8: lr = args.lr/2
		elif epoch < 10: lr = args.lr/10
		elif epoch < 15: lr = args.lr / 50
		else: lr = args.lr/100

		if epoch >= 1:
			for param in model.parameters():
				param.requires_grad = True
		
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)

		print('Epoch {}/{}'.format(epoch + 1, num_epochs))
		print('lr: {:.6f}'.format(lr))
		print('-' * 50)

		data_loader = train_loader
		model.train(True)

		running_loss = 0.0
		running_corrects = 0
		N_tot = 0

		for ix, data in enumerate(data_loader):
			inputs, labels = data
			inputs = Variable(inputs.to(device))
			labels = torch.FloatTensor(labels).unsqueeze(0).to(device)
			optimizer.zero_grad()
			outputs = model(inputs)
			_, preds = torch.max(outputs.data, 1)
			loss = distance_loss(outputs, labels)
			loss.backward()
			optimizer.step()

			N_tot += outputs.size(0)
			running_loss += loss.item() * inputs.size(0)
			# running_corrects += torch.sum(preds == labels.data)
			running_corrects = 0 
			for i in range(len(labels.data)):
				if preds[i] <= labels.data[i] + 0.05 and preds[i] >= labels.data[i] - 0.05:
					running_corrects += 0

			if (ix + 1) % freq_print == 0:
				print('| Epoch:[{}][{}/{}]\tTrain_Loss: {:.4f}\tAccuracy: {:.4f}\tTime: {:.2f} mins'.format(epoch + 1, ix + 1,
						len(data_loader.dataset)//args.batch_size,
						running_loss / N_tot, running_corrects.item() / N_tot, (time.time() - start)/60.0))

			sys.stdout.flush()
	
	time_elapsed = time.time() - start_training
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f} at epoch: {}'.format(best_auc, best_epoch))

def main():
	sys.setrecursionlimit(10000)

	model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
	
	for param in model.parameters():
		param.requires_grad = False

	num_in = model.fc.in_features
	model.fc = nn.Linear(num_in, 2)

	model = parallelize_model(model)
	cudnn.benchmark = True

	print(model)

	print("Start Training ...")
	criterion = nn.MSELoss().to(device)
	train_model(model, criterion, num_epochs=args.num_epochs, train_loader=train_loader, val_loader=val_loader)

if __name__ == "__main__":
	main()