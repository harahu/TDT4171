__author__ = 'kaiolae'
__author__ = 'kaiolae'
import Backprop_skeleton as Bp

#Class for holding your data - one object for each line in the dataset
class dataInstance:

    def __init__(self,qid,rating,features):
        self.qid = qid #ID of the query
        self.rating = rating #Rating of this site for this query
        self.features = features #The features of this query-site pair.

    def __str__(self):
        return "Datainstance - qid: "+ str(self.qid)+ ". rating: "+ str(self.rating)+ ". features: "+ str(self.features)


#A class that holds all the data in one of our sets (the training set or the testset)
class dataHolder:

    def __init__(self, dataset):
        self.dataset = self.loadData(dataset)

    def loadData(self,file):
        #Input: A file with the data.
        #Output: A dict mapping each query ID to the relevant documents, like this: dataset[queryID] = [dataInstance1, dataInstance2, ...]
        data = open(file)
        dataset = {}
        for line in data:
            #Extracting all the useful info from the line of data
            lineData = line.split()
            rating = int(lineData[0])
            qid = int(lineData[1].split(':')[1])
            features = []
            for elem in lineData[2:]:
                if '#docid' in elem: #We reached a comment. Line done.
                    break
                features.append(float(elem.split(':')[1]))
            #Creating a new data instance, inserting in the dict.
            di = dataInstance(qid,rating,features)
            if qid in dataset.keys():
                dataset[qid].append(di)
            else:
                dataset[qid]=[di]
        return dataset


def runRanker(trainingset, testset):
    #TODO: Insert the code for training and testing your ranker here.
    #Dataholders for training and testset
    dhTraining = dataHolder(trainingset)
    dhTesting = dataHolder(testset)

    #Creating an ANN instance - feel free to experiment with the learning rate (the third parameter).
    nn = Bp.NN(46,10,0.001)

    #TODO: The lists below should hold training patterns in this format: [(data1Features,data2Features), (data1Features,data3Features), ... , (dataNFeatures,dataMFeatures)]
    #TODO: The training set needs to have pairs ordered so the first item of the pair has a higher rating.
    trainingPatterns = [] #For holding all the training patterns we will feed the network
    testPatterns = [] #For holding all the test patterns we will feed the network
    for qid in dhTraining.dataset.keys():
        #This iterates through every query ID in our training set
        dataInstance=dhTraining.dataset[qid] #All data instances (query, features, rating) for query qid
        #TODO: Store the training instances into the trainingPatterns array. Remember to store them as pairs, where the first item is rated higher than the second.
        #TODO: Hint: A good first step to get the pair ordering right, is to sort the instances based on their rating for this query. (sort by x.rating for each x in dataInstance)
        dataInstance.sort(key=lambda instance: instance.rating)
        for i in range(len(dataInstance)):
            for pairing in dataInstance[i:]:
                if (dataInstance[i].rating != pairing.rating):
                    trainingPatterns.append((pairing.features, dataInstance[i].features))

    for qid in dhTesting.dataset.keys():
        #This iterates through every query ID in our test set
        dataInstance=dhTesting.dataset[qid]
        #TODO: Store the test instances into the testPatterns array, once again as pairs.
        #TODO: Hint: The testing will be easier for you if you also now order the pairs - it will make it easy to see if the ANN agrees with your ordering.
        dataInstance.sort(key=lambda instance: instance.rating)
        for i in range(len(dataInstance)):
            for pairing in dataInstance[i:]:
                if (dataInstance[i].rating != pairing.rating):
                    testPatterns.append((pairing.features, dataInstance[i].features))

    #Check ANN performance before training
    trainingProg = [0.0] * 26
    testingProg = [0.0] * 26
    for i in range(10):
        nn = Bp.NN(46,10,0.0005)
        trainingProg[0] += (1-nn.countMisorderedPairs(trainingPatterns))/10.0
        testingProg[0] += (1-nn.countMisorderedPairs(testPatterns))/10.0
        for j in range(25):
            #Running 25 iterations, measuring testing performance after each round of training.
            #Training
            nn.train(trainingPatterns,iterations=1)
            #Check ANN performance after training.
            trainingProg[j+1] += (1-nn.countMisorderedPairs(trainingPatterns))/10.0
            testingProg[j+1] += (1-nn.countMisorderedPairs(testPatterns))/10.0
            print("Run: "+str(i))
            print(trainingProg)
            print(testingProg)

    #TODO: Store the data returned by countMisorderedPairs and plot it, showing how training and testing errors develop.



runRanker("train.txt","test.txt")
