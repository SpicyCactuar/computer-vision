import numpy as np
from numpy import genfromtxt
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import cv2
import os

#files = glob.glob("./resources/train_dataset/*.csv")
perturbed_files = glob.glob("./resources/train_dataset_big/*perturbed*")
original_files = glob.glob("./resources/train_dataset_big/*original*")
label_files = glob.glob("./resources/train_dataset_big/*.csv")
if len(perturbed_files) != len(original_files):
	print("Perturbed/Original mismatch")
if len(perturbed_files) != len(label_files):
	print("Image/Label mismatch")

data = [[] for x in range(8)]

i = 0
for filename in perturbed_files:
	perturbed = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	if np.all(perturbed == perturbed[0,0]):
		print("All the same at: " + filename)
		print("Being " + str(perturbed[0,0]))
		original = cv2.imread(filename.replace("perturbed", "original"), cv2.IMREAD_GRAYSCALE)
		if np.all(original == perturbed[0,0]):
			print("And the original too")
			print("Removing...")
			os.remove(filename)
			os.remove(filename.replace("perturbed", "original"))
			os.remove(filename.replace("_perturbed.png", ".csv"))
			continue

	# label_array = genfromtxt(label, delimiter=",") / 16.0
	# for j in range(8):
		# data[j].append(label_array[j])
	i += 1
	if i%1000 == 0:
		print(i)

# print("Finished loading data, plotting...")
#
# for j in range(8):
# 	sns.distplot(data[j]);
# 	plt.show()
#
# print data
