import csv
from scipy.sparse import lil_matrix
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

#The most intuitive way to do so is the bags of words representation:
#1. assign a fixed integer id to each word occurring in any document of the training set 
#(for instance by building a dictionary from words to integer indices).
#2. for each document #i, count the number of occurrences of each word w and store it in 
#X[i, j] as the value of feature #j where j is the index of word w in the dictionary

#In a given row:
#0 - Reference ID
#1 - Report Year
#2 - Diagnosis Category
#3 - Diagnosis Sub Category
#4 - Treatment Category
#5 - Treatment Sub Category
#6 - Determination
#7 - Type
#8 - Age Range
#9 - Patient Gender
#10 - Severity
#11 - Days to Review
#12 - Days to Adopt
#13 (and everything onwards) - Findings

stopWords = ['a', 'about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 'became', 'because', 'become', 'becomes', 'been', 'before', 'began', 'behind', 'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 'came', 'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'come', 'could', 'd', 'did', 'differ', 'different', 'differently', 'do', 'does', 'done', 'down', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'either', 'end', 'ended', 'ending', 'ends', 'enough', 'even', 'evenly', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'herself', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'important', 'in', 'interest', 'interested', 'interesting', 'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows', 'l', 'large', 'largely', 'last', 'later', 'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long', 'longer', 'longest', 'm', 'made', 'make', 'making', 'man', 'many', 'may', 'me', 'member', 'members', 'men', 'might', 'more', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'n', 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'new', 'newer', 'newest', 'next', 'no', 'nobody', 'non', 'noone', 'not', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on', 'once', 'one', 'only', 'open', 'opened', 'opening', 'opens', 'or', 'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'our', 'out', 'over', 'p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point', 'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 'really', 'right', 'rights', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems', 'sees', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 'shows', 'side', 'sides', 'since', 'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states', 'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things', 'think', 'thinks', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 'through', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'turn', 'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 'upon', 'us', 'use', 'used', 'uses', 'v', 'very', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 'what', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whose', 'why', 'will', 'with', 'within', 'without', 'work', 'worked', 'working', 'works', 'would', 'x', 'y', 'year', 'years', 'yet', 'you', 'young', 'younger', 'youngest', 'your', 'yours', 'z']
stopWords = set(stopWords)
def getAllWords(directory, arrayOfFileNames):
	words = {}
	for filename in arrayOfFileNames:
		filename = directory + filename
		with open(filename, 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	
			#reads contents of file into a vector where each entry represents a single patient
			patients = []
			for row in spamreader:
				patients.append(row)
			#creates dictionary where keys are words occurring in any patient and values are unique IDs for those words
			idTracker = 0
			count = 0
			for patient in patients:
					if patient[2] == 'Mental Disorder':
						count += 1
						#starts at 3 because fields 0, 1, and 2 are not relevant
						for i in range(3, len(patient)):
							currentPhrase = (patient[i]).split()
							for word in currentPhrase:
								word = word.lower()
								word = "".join(c for c in word if c not in ('\'', '/', '!', '(', ')', '.', ':', ';', '?'))
								if not word in stopWords:
									if not word in words:
										words[word] = idTracker
										idTracker += 1
	listwords = list(words)
	mapWordToIndex = {listwords[i]: i for i in range(len(listwords))}
	return mapWordToIndex

mapWordToIndex = getAllWords('/Users/juanm95/Documents/cs221/project/ChatMD/', ['dev.csv', 'test.csv'])

def getFeaturesAndClassifications(filename):
	with open(filename, 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		
		#reads contents of file into a vector where each entry represents a single patient
		patients = []
		for row in spamreader:
			patients.append(row)
		#creates dictionary where keys are words occurring in any patient and values are unique IDs for those words
		# idTracker = 0
		# count = 0
		# for patient in patients:
		# 	if patient[2] == 'Mental Disorder':
		# 		count += 1
		# 		#starts at 3 because fields 0, 1, and 2 are not relevant
		# 		for i in range(3, len(patient)):
		# 			currentPhrase = (patient[i]).split()
		# 			for word in currentPhrase:
		# 				word = word.lower()
		# 				word = "".join(c for c in word if c not in ('\'', '/', '!', '(', ')', '.', ':', ';', '?'))
		# 				if not word in stopWords:
		# 					if not word in words:
		# 						words[word] = idTracker
		# 						idTracker += 1

		#counts the number of occurrences of each word and stores it in data[i,j] wherein
			#i is the patient number and j is the index of word w in the dictionary
		data = []
		for i in range(len([patient for patient in patients if patient[2] == "Mental Disorder"])):
			data.append([0] * len(mapWordToIndex))
		# data = lil_matrix((count, idTracker))
		patientID = 0
		disorders = set()
		patientSet = set()
		yVector = []
		for patient in patients:
			if patient[2] == 'Mental Disorder':
				#stores number of occurrences of each word in counts wherein
					#the word is the key and the value is the number of occurrences
				disorders.add(patient[3])
				counts = {}
				for i in range(3, len(patient)):
					currentPhrase = (patient[i]).split()
					for word in currentPhrase:
						word = word.lower()
						word = "".join(c for c in word if c not in ('\'', '/', '!', '(', ')', '.', ':', ';', '?'))
						if not word in stopWords:
							if not word in counts:
								counts[word] = 1
							else:
								counts[word] += 1
				#stores number of occurrences of each word in data
				for word in counts:
					patientSet.add(patientID)
					data[patientID][mapWordToIndex[word]] = counts[word]
				yVector.append(patient[3])
				patientID += 1
		disorderMap = list(disorders)
		mapToTheIndex = {disorderMap[i]: i for i in range(len(disorderMap))}
		for i in range(len(yVector)):
			yVector[i] = mapToTheIndex[yVector[i]]
		y = np.array(yVector)
		X = np.array(data)
	return (X, y, disorderMap)
		
X, y, disorderMap = getFeaturesAndClassifications('/Users/juanm95/Documents/cs221/project/ChatMD/dev.csv')	
print "Read in %s patients" % len(X)
classifier = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y)

def textToMatrix(text):
	data = [[0] * len(mapWordToIndex)]
	counts = {}
	for word in text.split():
		word = word.lower()
		word = "".join(c for c in word if c not in ('\'', '/', '!', '(', ')', '.', ':', ';', '?'))
		if not word in stopWords:
			if word not in counts:
				counts[word] = 0
			counts[word] += 1
	#stores number of occurrences of each word in data
	for word in counts:
		if word in mapWordToIndex:
			data[0][mapWordToIndex[word]] = counts[word]
	return np.array(data)

while True:
	textToClassify = raw_input("Input Text to Classify or enter nothing to quit\n")
	if textToClassify == "":
		break
	matrixToClassify = textToMatrix(textToClassify)
	classification = classifier.predict(matrixToClassify)
	print disorderMap[classification[0]]

# testX, testY = getFeaturesAndClassifications('/Users/juanm95/Documents/cs221/project/test.csv')	
# results = classifier.predict(testX)
# count = 0
# for i in range(len(results)):
# 	if testY[i] != results[i]:
# 		count += 1
# print float(count) / len(results)
