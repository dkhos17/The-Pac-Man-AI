# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util, sys
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        "*** YOUR CODE HERE ***"
        best = (0, self.weights.copy()) # tuple of (best_guesses, best_weights)
        init_weights = self.weights.copy() # save initial weights for iterations.
        for c in Cgrid:
            for _ in range(self.max_iterations):
                for i,F in enumerate(trainingData):
                    F = trainingData[i].copy() # get f vector
                    realY = trainingLabels[i] # get y(real label) for given f vector

                    # get y'(current y) according to current weights, y' = argmax(score(f,y''))
                    currY = self.classify([F])[0]
                    # if y' == y, do nothing (weights is ok.)
                    if currY == realY: continue

                    tau = min(c, ((self.weights[currY]-self.weights[realY]) * F + 1.0) / (2.0*(F*F)))
                    # update weights - w[y']= w[y'] - tau*f, w[y] = w[y] + tau*f
                    F.divideAll(1.0/tau) # don't calculate this twice and use divideAll as multiply all.
                    self.weights[realY] += F
                    self.weights[currY] -= F
                
            # count how much we will guess if weights where this. 
            curr_guesses = util.Counter()
            curr_guesses.incrementAll([currLabel == validationLabels[i] for i, currLabel in enumerate(self.classify(validationData))],1)
            
            # if we have better result update and save the best result
            best = max(best, (curr_guesses[True], self.weights.copy()))
            self.weights = init_weights.copy() # init self.weights again.

        
        # set self.weights with the best weight we've counted.
        self.weights = best[1]

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


# SCR = util.Counter() # save scores for each y' on f
# # for loop on each y'
# for tmpY, W in self.weights.items():
#     # score = E(sum over i)f[i]*w[i] = f*w
#     SCR[tmpY] = F*W
# currY = SCR.argMax()
