import SimpleITK as sitk
import numpy as np
import csv
import os
import re
import random
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure, morphology
import scipy
import scipy.misc
from keras.models import model_from_json

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class DataHandler(object):
	def __init__(self):
		pass


	def load_itk_image(self, filename):
		'''
		Takes in a string to the file location
		Returns:
			numpyImage: a numpy array of the image raw data
			numpyOrigin:
			numpySpacing: the spacing conversion between voxels in the
				x, y and z direction and real world lengths
		'''
		itkimage = sitk.ReadImage(filename)
		numpyImage = sitk.GetArrayFromImage(itkimage)

		numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
		numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

		return numpyImage, numpyOrigin, numpySpacing

	def readCSV(self, filename):
		'''
		Takes in a string to the CSV file location
		Returns a list of lines in the csv file
		'''
		lines = []
		with open(filename, "rb") as f:
			csvreader = csv.reader(f)
			for line in csvreader:
				lines.append(line)
		return lines

	def worldToVoxelCoord(self, worldCoord, origin, spacing):
		'''
		Converts a distance in world coordinates to voxel coordinates
		'''

		stretchedVoxelCoord = np.absolute(worldCoord-origin)
		voxelCoord = stretchedVoxelCoord / spacing
		return voxelCoord

	def normalizePlanes(self, npzarray):
		maxHU = 400.0
		minHU = -1000.0
		npzarray = (npzarray-minHU) / (maxHU - minHU)
		npzarray[npzarray>1] = 1
		npzarray[npzarray<0] = 0
		return npzarray

	def generateSamples(self, imageDir, csv, savedir, targetWorldWidth, targetVoxelWidth):
		#get a list of files and candidates
		mhd = re.compile(r".*\.mhd")
		files = [f for f in os.listdir(imageDir) if mhd.match(f) != None]
		cands = self.readCSV(csv)

		count = {'0':0, '1':0}

		#organize candidates into a dictionary with patient id as the key
		candDict = {}
		for cand in cands[1:]:
			if not candDict.has_key(cand[0]):
				candDict[cand[0]] = []
			candDict[cand[0]].append(cand)

		candDictVoxel = {}
		#extract candidates
		count2 = 0
		for f in reversed(files):
			trueNodules = np.zeros((1, targetVoxelWidth[0], targetVoxelWidth[1], targetVoxelWidth[2]))
			falseCandidates =  np.zeros((1, targetVoxelWidth[0], targetVoxelWidth[1], targetVoxelWidth[2]))
			randomSamples = np.zeros((1, targetVoxelWidth[0], targetVoxelWidth[1], targetVoxelWidth[2]))
			print("Extracting from file {}".format(f))
			print("Candidates")

			if candDict.has_key(f[0:-4]): #if the patient has no candidates, skip
				img, origin, spacing = self.load_itk_image("{}{}".format(imageDir, f))  #load image
				voxelWidth = targetWorldWidth / spacing  #calculate the width of the box to extract

				#extract each candidate in patient f
				for cand in candDict[f[0:-4]]:

					worldCoord = np.asarray([float(cand[3]), float(cand[2]), float(cand[1])])
					voxelCoord = self.worldToVoxelCoord(worldCoord, origin, spacing)

					#make a dictionary of candidates and their voxel coordinates for
					#use later in random sample generation
					if not candDictVoxel.has_key(cand[0]):
						candDictVoxel[cand[0]] = []
					candDictVoxel[cand[0]] += [voxelCoord]

					count[cand[4]] += 1 #count the number of candidates that are true nodules and are nt
					patch = self.extractPatch(voxelCoord, voxelWidth, img) #extract the patch

					#for z in range(patch.shape[0]):
					#	plt.imshow(patch[z], cmap='gray')
						#plt.show()

					#resize the patch to targetVoxelWidth
					if(patch.shape[0] != 0) and (patch.shape[1] != 0) and (patch.shape[2] != 0):
						count2 += 1
						patch = scipy.ndimage.interpolation.zoom(patch, (float(targetVoxelWidth[0])/patch.shape[0],
																	float(targetVoxelWidth[1])/patch.shape[1],
																	float(targetVoxelWidth[2])/patch.shape[2]))

						#save sample
						patch = patch.reshape((1, targetVoxelWidth[0], targetVoxelWidth[1], targetVoxelWidth[2]))
						if cand[4] == '1':
							trueNodules = np.concatenate((patch, trueNodules))
						else:
							falseCandidates = np.concatenate((patch, falseCandidates))
						#np.ndarray.tofile(patch, "{}{}/{}-{}".format(savedir, cand[4], cand[0], count[cand[4]]))


				#extract random samples
				print("Random")
				for i in range(500):
					#generate random coordinates
					coord = np.array([int(random.uniform(voxelWidth[0], img.shape[0]-voxelWidth[0])),
									int(random.uniform(voxelWidth[1], img.shape[1]-voxelWidth[1])),
									int(random.uniform(voxelWidth[2], img.shape[2]-voxelWidth[2]))
						])
					bad = False
					low = coord - voxelWidth
					high = coord + voxelWidth

					#check if the coordinates conflict with any known candidates
					for cand in candDictVoxel[f[0:-4]]:
						if (low[0]<cand[0]<high[0]) and (low[1]<cand[1]<high[1]) and (low[2]<cand[2]<high[2]):
							bad = True
							break

					#if not, then extract the sample, resize, and save
					if not bad:
						count2 += 1
						patch = self.extractPatch(coord, voxelWidth, img)
						patch = scipy.ndimage.interpolation.zoom(patch, (float(targetVoxelWidth[0])/patch.shape[0],
																	float(targetVoxelWidth[1])/patch.shape[1],
																	float(targetVoxelWidth[2])/patch.shape[2]))
						patch = patch.reshape((1, targetVoxelWidth[0], targetVoxelWidth[1], targetVoxelWidth[2]))
						randomSamples = np.concatenate((patch, randomSamples))
						#np.ndarray.tofile(patch, "{}{}/{}-{}".format(savedir, 2, f[0:-4], i))
			if trueNodules.shape[0] >1: np.ndarray.tofile(trueNodules[0:-1], "{}/1/{}".format(savedir, f[0:-4]))
			if falseCandidates.shape[0] >1: np.ndarray.tofile(falseCandidates[0:-1], "{}/0/{}".format(savedir, f[0:-4]))
			if randomSamples.shape[0] >1: np.ndarray.tofile(randomSamples[0:-1], "{}/2/{}".format(savedir, f[0:-4]))
		print(count)
		print(count2)

	def load_samples(self, sampleDir, shape):
		size = shape[1]*shape[2]*shape[3]

		print("Loading false nodules")
		falseNodules = [np.fromfile("{}0/{}".format(sampleDir, f)) for f in os.listdir("{}0/".format(sampleDir))]
		falseNoduleList = []
		for f in falseNodules:
			falseNoduleList += list(f.reshape((f.shape[0]/size, shape[0], shape[1], shape[2], shape[3])))
		falseNoduleList = zip(falseNoduleList, [1]*len(falseNoduleList))

		print("Loading true nodules")
		trueNodules = [np.fromfile("{}1/{}".format(sampleDir, f)) for f in os.listdir("{}1/".format(sampleDir))]
		trueNoduleList = []
		for f in trueNodules:
			trueNoduleList += list(f.reshape((f.shape[0]/size, shape[0], shape[1], shape[2], shape[3])))
		trueNoduleList = zip(trueNoduleList, [1]*len(trueNoduleList))

		print("Loading random samples")
		randomSamples = [np.fromfile("{}2/{}".format(sampleDir, f)) for f in os.listdir("{}2/".format(sampleDir))]
		randomSampleList = []
		for f in randomSamples:
			randomSampleList += list(f.reshape((f.shape[0]/size, shape[0], shape[1], shape[2], shape[3])))
		randomSampleList = zip(randomSampleList, [0]*len(randomSampleList))
		print(len(randomSampleList))

		samples = trueNoduleList + falseNoduleList + randomSampleList
		random.shuffle(samples)
		xs = [x[0] for x in samples]
		ys = [y[1] for y in samples]
		return np.asarray(xs), np.asarray(ys)

	def saveModel(self, mod, targetDir, name):
	    #serialize model to JSON
	    model_json = mod.to_json()
	    with open("{}{}.json".format(targetDir,name), "w") as json_file:
	        json_file.write(model_json)
	    #serialize weights to HDF5
	    mod.save_weights("{}{}.h5".format(targetDir,name))
	    print("Model saved to {}{}".format(targetDir,name))

	def loadModel(self, sourceDir, name):
	    json_file = open("{}{}.json".format(sourceDir,name))
	    loaded_model_json = json_file.read()
	    json_file.close()
	    loaded_model = model_from_json(loaded_model_json)
	    # load weights into new model
	    loaded_model.load_weights("{}{}.h5".format(sourceDir,name))
	    print("Loaded model from {}{}".format(sourceDir,name))
	    return loaded_model

	def extractPatch(self, voxelCoord, voxelWidth, numpyImage):
		patch = numpyImage[int(voxelCoord[0]-voxelWidth[0]/2):int(voxelCoord[0]+voxelWidth[0]/2),
							int(voxelCoord[1]-voxelWidth[1]/2):int(voxelCoord[1]+voxelWidth[1]/2),
							int(voxelCoord[2]-voxelWidth[2]/2):int(voxelCoord[2]+voxelWidth[2]/2)]
		patch = self.normalizePlanes(patch)
		return patch

	def show_3d_img(self, img):
		for z in range(img.shape[0]):
			plt.imshow(img[z], cmap='gray')
			plt.show()

	def save_slices(self, img, stride, directory):
		for i in range(0, img.shape[0], stride):
			scipy.misc.imsave('{}{}.jpg'.format(directory, i), img[i])


def main():
	handler = DataHandler()

	'''
	shape = {}
	spacingg = {}
	mhd = re.compile(r".*\.mhd")
	files = [f for f in os.listdir("/media/amos/My Passport/Data/luna/subset0/") if mhd.match(f) != None]

	z = []
	x = []
	y = []

	for f in files:
		print(f)
		img, origin, spacing = handler.load_itk_image("/media/amos/My Passport/Data/luna/subset0/{}".format(f))
		if not shape.has_key(str(img.shape)):
			shape[str(img.shape)] = 0;
		if not spacingg.has_key(str(spacing)):
			spacingg[str(spacing)] = 0;
		shape[str(img.shape)] += 1;
		spacingg[str(spacing)] += 1;
		z += [spacing[0]]
		x += [spacing[1]]
		y += [spacing[2]]
	'''








	for i in range(8,10):
		print("doing {}".format(i))
		handler.generateSamples("/media/amos/My Passport/Data/luna/subset{}/".format(i), "data/csv/candidates.csv", "sampless/subset{}/".format(i), (10,10,10), (10,16,16))
	xs, ys = handler.load_samples("sampless/subset0/", (1, 10,16,16))
	print(xs.shape)
	print(ys.shape)


	#for z in range(xs.shape[2]):
		#plt.imshow(xs[0][0][z], cmap='gray')
		#plt.show()



if __name__ == "__main__":
	main()
