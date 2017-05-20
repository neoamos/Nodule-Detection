import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import MaxPooling3D
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json
from FCN import FCN
import random
import os
from Data import DataHandler
import scipy
import re
import csv
from math import floor


#Takes an array with probabilities and indicies for these probabilities.
#Returns that array with all overlapping indicies removed


#1 represents not deleted
def non_max_suppression(array, indicies):
	#Sorts the list of indicies by probability values
	indicies = sorted(indicies, key=lambda x: x[1])
	#Reverses the list so that the first values are the largest
	indicies = list(reversed(indicies))

	#Used to store all of the deleted indicies
	deleted_indicies = {}

	for index in indicies:
		deleted_indicies[index[0]] = 1

	i = 0
	#Goes through each index
	wz = 8
	wx = 12
	wy = 12
	keptIndicies = []
	mask = np.zeros((wz*2,wx*2,wy*2))
	mask[1][2][2] = 1
	for index in range(len(indicies)):

		zi = indicies[index][0][0]
		xi = indicies[index][0][1]
		yi = indicies[index][0][2]

		if array[zi][xi][yi] != 0:
			keptIndicies += [indicies[index]]
			if (zi+wz < array.shape[0]) and (zi-wz>0) and (xi+wx < array.shape[1]) and (xi-wx>0) and (yi+wy < array.shape[2]) and (yi-wy >0):
				array[zi-wz:zi+wz, xi-wx:xi+wx, yi-wy:yi+wy] = mask
			else:
				for z in range (zi-wz, zi+wz):
					for x in range (xi-wx, xi+wx):
						for y in range (yi-wy, yi+wy):
							if (0< z < array.shape[0]) and (0< x < array.shape[1]) and (0< y < array.shape[2]):
								array[z][x][y] = 0;
				array[zi][xi][yi] = 1

	return array, keptIndicies


def getCandidates(imageDir, FCN):
    mhd = re.compile(r".*\.mhd")
    files = [f for f in os.listdir(imageDir) if mhd.match(f) != None]
    savedOut = [f[0:-4] for f in os.listdir("../Data/savedFCNOut/")]
    imageNames = []
    keptIndiciesList = []
    for f in files:
		image, orig, spacing = handler.load_itk_image("{}{}".format(imageDir, f))
		print("Analyzing {}".format(f[0:-4]))
		if(savedOut.__contains__(f[0:-4])):
		    pred = np.load("../Data/savedFCNOut/{}.npy".format(f[0:-4]))
		else:
		    pred, inputSize = FCN.predict(image, spacing, (1, 0.625, 0.625))
		    np.save("../Data/savedFCNOut/{}".format(f[0:-4]), pred)

		print("FCN output size: {}".format(pred.shape))
		#pred = np.load("fullresnorm4.npy")
		#reshape
		#pred = pred.reshape(1, *pred.shape)
		yyes = pred[0, :, :, :, 1].reshape(1, pred.shape[1], pred.shape[2], pred.shape[3])
		ynot = pred[0, :, :, :, 0].reshape(1, pred.shape[1], pred.shape[2], pred.shape[3])
		pred = np.concatenate((ynot,yyes))
		pred = pred.reshape(1,2,pred.shape[1], pred.shape[2],pred.shape[3])
		#handler.show_3d_img(pred[0][1])
		inputSize = (int(image.shape[0]*float(spacing[0])),
		            int(image.shape[1]*float(spacing[1])/0.625),
		            int(image.shape[2]*float(spacing[2])/0.625))
		print("input size: {}".format(inputSize))
		populatedArray, indicies = FCN.indexMapping(pred, inputSize)
		print("{} indicies found".format(len(indicies)))
		removedOverlap, keptIndicies = non_max_suppression(populatedArray,indicies)
		print("{} indicies kept".format(len(keptIndicies)))
		keptIndiciesList += [keptIndicies]
		imageNames += [f[0:-4]]
		#generateStats(imageNames, keptIndiciesList)

    return imageNames, keptIndiciesList

#Reads in a CSV file
#Changed to return a dictionary where you can reference all coordinate sets based upon the file name
#Reads in a CSV file
def readCSV(filename):
		'''
		Takes in a string to the CSV file location
		Returns a list of lines in the csv file
		'''
		lines = []
		with open(filename, "rb") as f:
			csvreader = csv.reader(f)
			for line in csvreader:
				lines.append(line)

		#Contains the values
		image = {}
		#Converts the list into a hashmap
		for line in lines:
			if line[0] in image:
				value = image[line[0]]
				value.append((line[3], line[2], line[1], line[4]))
				image[line[0]] = value
			else:
				value = [(line[3], line[2], line[1], line[4])]
				image[line[0]] = value

		return image

#Takes in a tuple of image names and their generated indicies, returns aggregate statistics as well as particular ones
def generateStats(names, generatedIndiciesList):
	#Reads in the CSV
	image = readCSV("../Data/data/csv/candidates.csv")
	#Variables that represent aggregates
	aggTruePositives = 0
	aggNumPositives = 0
	aggFalsePositives = 0
	aggNumGuesses = 0
	aggFalseNegatives = 0
	aggNodules = 0
	aggNodulesFound = 0

	for i in range(len(names)):
		print("Statistics for the " + str(i + 1) + "th image")
		points = image[names[i]]
		generatedIndicies = generatedIndiciesList[i]
		truePositives, numPositives, falsePositives, falseNegatives, nodules, nodulesFound = checkAccuracy(generatedIndicies, image,names[i])
		print("True Positives --> " + str(truePositives) + " Out of ---> " + str(numPositives) + " Positives found % --> " + str(float(truePositives)/float(numPositives)))
		falsePositivePct = float(falsePositives) / float(len(generatedIndicies))
		print("False Positives Found --> " + str(falsePositives) + " Out of " + str(len(generatedIndicies)) + " For a % --> " + str(falsePositivePct))
		fnPct = float(falseNegatives)/ float(numPositives)
		print("False Negatives Found --> " + str(falseNegatives) + " With a % --> " + str(fnPct))

		print("Out of : " + str(nodules) + " We found: "+ str(nodulesFound) + " For a Nodules Found % --> " +  str(float(nodulesFound)/(float(nodules)+0.01)))

		aggTruePositives = aggTruePositives + truePositives
		aggNumPositives = aggNumPositives + numPositives
		aggFalsePositives = aggFalsePositives + falsePositives
		aggNumGuesses = aggNumGuesses + len(generatedIndicies)
		aggFalseNegatives = aggFalseNegatives + falseNegatives
		aggNodules = aggNodules + nodules
		aggNodulesFound = aggNodulesFound + nodulesFound

	print("******* HERE ARE THE AGGREGATE RESULTS *******")
	print("--- True Positives ---")
	print("Found --> " + str(aggTruePositives))
	print("Out of --> " + str(aggNumPositives))
	print("Percent --> " + str(float(aggTruePositives)/ float(aggNumPositives)))
	print("--- False Positives ---")
	print("Found --> " + str(aggFalsePositives))
	print("Out of --> " + str(aggNumGuesses))
	print("Percent --> " + str(float(aggFalsePositives)/ float(aggNumGuesses)))
	print("--- Missed Positives ---")
	print("Found --> " + str(aggFalseNegatives))
	print("Out of --> " + str(aggNumPositives))
	print("Percent --> "+ str(float(aggFalseNegatives)/ float(aggNumPositives)))
	print("--- Nodules ---")
	print("Found --> " + str(aggNodulesFound))
	print("Out of --> " + str(aggNodules))
	print("Percent --> " + str(float(aggNodulesFound)/ float(aggNodules)))

def checkAccuracy(generatedIndicies, image, filename):
	truePositives = 0
	falsePositives = 0
	falseNegatives = 0
	noduleCount = 0
	nodulesFound = 0

	#Generates the origin and spacing values for the image
	handler = DataHandler()
	filepath = "../Data/toPredict/" + filename + ".mhd"
	_, origin, spacing = handler.load_itk_image(filepath)

	#Generates the list of indicies that we need for the file
	points = image[filename]
	correctIndicies = {}
	for point in points:
		#Converts values in points to voxel and scales them correctly
		worldCoords = np.asarray([float(point[0]), float(point[1]), float(point[2])])
		#print(worldCoords)

		voxelCoords = handler.worldToVoxelCoord(worldCoords, origin, spacing)
		#print(voxelCoords)
		#Scales for the resolution that we want
		targetSpacing = np.array([1, .625, .625])
		multiplier = spacing / targetSpacing
		#print("-----")
		#print(voxelCoords)
		voxelCoords = voxelCoords * multiplier
		#print(voxelCoords)
		#print("-----")
		valuesTuple = (tuple(voxelCoords), int(point[3]))
		correctIndicies[valuesTuple] = 0

	#Goes through each guessed index and figures out of it is false or not
	for guess in generatedIndicies:
		#Goes through each correct answer

		isCorrect = False
		for correct in correctIndicies:
			#Guess coordinates you are judging based off of
			guessZ = guess[0][0]
			guessX = guess[0][1]
			guessY = guess[0][2]

			correctZ = correct[0][0]
			correctX = correct[0][1]
			correctY = correct[0][2]

			if (guessZ - 5 < correctZ) and (guessZ + 5 > correctZ) and (guessX - 8 < correctX) and (guessX + 8 > correctX) and (guessY - 8 < correctY) and (guessY + 8 > correctY):
				correctIndicies[correct] = 1
				isCorrect = True

		if isCorrect == False:
			falsePositives = falsePositives + 1

		#If the guess was not any correct answer then it is a false positive
		#if not isCorrect:
		#	falsePositives += 1

	for ind in correctIndicies:
		if ind[1] == 1:
			noduleCount = noduleCount + 1
			if correctIndicies[ind] == 1:
				nodulesFound = nodulesFound + 1
		#If we found it
		if correctIndicies[ind] == 1:
			truePositives = truePositives + 1
		if correctIndicies[ind] == 0:
			falseNegatives = falseNegatives + 1


	numPositives = len(correctIndicies)
	return truePositives, numPositives, falsePositives, falseNegatives, noduleCount, nodulesFound


handler = DataHandler()
FCN = FCN(2, "../Data/model/", "model/8")
imageNames, keptIndiciesList = getCandidates("../Data/toPredict/", FCN)
#print(keptIndiciesList)
generateStats(imageNames, keptIndiciesList)

'''
y , orig, spacing = handler.load_itk_image("../Data/data/1.3.6.1.4.1.14519.5.2.1.6279.6001.220596530836092324070084384692.mhd")
#pred = predFull(FCN, y, spacing, (1, 0.625, 0.625))
#print(pred.shape)
cand = np.load("../Data/fullresnorm5.npy")
print(cand.shape)
handler.save_slices(y, 10, "../Data/images/CTscan/")

xs, ys = handler.load_samples("../Data/sampless/subset{}/".format(i), (10,16,16,1))
print(xs.shape)
print(ys.shape)
trainSize = int(xs.shape[0]*0.2)

print("training on {} samples".format(trainSize))

X_train = xs[0:-trainSize]
y_train = ys[0:-trainSize]
X_test = xs[-trainSize:]
y_test = ys[-trainSize:]

print(X_train.shape)
for i in range(0, 5):
    if y_train[i] == 1:
        handler.save_slices(X_train[i][:,:,:,0], 1, "../Data/images/{}-".format(i))
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# one hot encode outputs
#Turns each value into a 1 at the correct position of a vector
#EX 6 becomes 1 X 10 Vector --> [0,0,0,0,0,1,0,0,0,0]
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_train = y_train.reshape(y_train.shape[0],1,1,1,2)
y_test = y_test.reshape(y_test.shape[0],1,1,1,2)

#Trains the model you made on the X and y training data set
#Uses X_test and y_test to check accuracy while it is training
#FCN.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32)
# Final evaluation of the model
#scores = FCN.evaluate(X_test, y_test, verbose=0)
#print(scores)
#saveModel(FCN, "model/", "10")
#FCN = buildFCN("8")
handler= DataHandler()
y , orig, spacing = handler.load_itk_image("../Data/data/1.3.6.1.4.1.14519.5.2.1.6279.6001.220596530836092324070084384692.mhd")
print(orig)
print(y.shape)
pred = predict(FCN, y[0:25], spacing, (1, 0.625, 0.625))
np.save("../Data/npred", pred)
pred1 = predict(FCN, y[48:100], spacing, (1, 0.625, 0.625))
np.save("../Data/npred1", pred1)
pred2 = predict(FCN, y[98:150], spacing, (1, 0.625, 0.625))
np.save("../Data/npred2", pred2)
pred3 = predict(FCN, y[148:200], spacing, (1, 0.625, 0.625))
np.save("../Data/npred3", pred3)
pred4 = predict(FCN, y[198:250], spacing, (1, 0.625, 0.625))
np.save("../Data/npred4", pred4)
pred5 = predict(FCN, y[248:300], spacing, (1, 0.625, 0.625))
np.save("../Data/npred5", pred5)
pred6 = predict(FCN, y[298:-1], spacing, (1, 0.625, 0.625))
np.save("../Data/npred6", pred6)
pred = np.load("../Data/npred.npy")
pred1 = np.load("../Data/npred1.npy")
pred2 = np.load("../Data/npred2.npy")
pred3 = np.load("../Data/npred3.npy")
pred4 = np.load("../Data/npred4.npy")
pred5 = np.load("../Data/npred5.npy")
pred6 = np.load("../Data/npred6.npy")


pred = np.concatenate((pred[0][0], pred1[0][0], pred2[0][0], pred3[0][0], pred4[0][0], pred5[0][0], pred6[0][0]))
#pred2 =predict(FCN, y[50:-1], spacing, (1, 0.625, 0.625))
#pred = np.concatenate((pred6[0], pred5[0], pred4[0], pred3[0], pred2[0], pred1[0], pred[0]))
#np.save("cand3", pred)
#print(pred.shape)
#preds = pred[:, :, :, 0]
#print(preds.shape)
#cand = np.load("normcand.npy")
#print(cand.shape)
#cand = np.concatenate((cand[0:21], cand[21:42], cand[42:63], cand[63:84], cand[84:105], cand[105:126], cand[126:-1]))
#cand = np.flip(np.concatenate((np.flip(cand[126:-1], 0), np.flip(cand[105:126], 0), np.flip(cand[84:126], 0), np.flip(cand[63:84], 0), np.flip(cand[42:63], 0),
        #np.flip(cand[21:42], 0), np.flip(cand[0:21], 0))), 0)
np.save("../Data/fullresnorm5", pred)
print(pred.shape)
#print(cand2.shape)
handler.show_3d_img(pred[:, :, :, 1])
'''
