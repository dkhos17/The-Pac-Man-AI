    # analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

### if we need to get the highest reward way,
## we need to have high probabilty  
# to make the move what the agent have done,
# than the agent will go the 'risky' way bridge. 
# this happens when noise is low..
def question2():
    answerDiscount = 0.9
    answerNoise = 0.002
    return answerDiscount, answerNoise


### if we need to get the (a) way,
## we need to have high probabilty  
# to make the move what the agent have done,
# than the agent will go the 'risky' cliff.
# and if we want to choose the low reward(+1) way
# we need this: 1*gama^3 > 10*gama^5
# we dont need living reward, else we 'never' finish  
def question3a():
    answerDiscount = 0.2
    answerNoise = 0.002
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


### if we need to get the (b) way,
## we need to have high noise probabilty  
# to avoid the the risky cliff,
# than the agent will go the 'unrisky' cliff.
# and if we want to choose the low reward(+1) way
# we need this: 1*gama^7 > 10*gama^9
# we dont need living reward, else we 'never' finish
def question3b():
    answerDiscount = 0.3
    answerNoise = 0.2
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


### if we need to get the (c) way,
## we need to have high probabilty  
# to make the move what the agent have done,
# than the agent will go the 'risky' cliff.
# and if we want to choose the high reward(+10) way
# we need this: 1*gama^3 < 10*gama^5 
# we dont need living reward, else we 'never' finish
def question3c():
    answerDiscount = 0.9
    answerNoise = 0.002
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

### if we need to get the (d) way,
## we need to have high noise probabilty  
# to avoid the the risky cliff,
# than the agent will go the 'unrisky' cliff.
# and if we want to choose the low reward(+1) way
# we need this: 1*gama^7 < 10*gama^9
# we dont need living reward, else we 'never' finish
def question3d():
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

### if we want to never terminate an episode
# the agent should not wanted to go terminal and finish the game
# if we have a positive reward for living, the agent will never try to finish game
def question3e():
    answerDiscount = 0.9
    answerNoise = 0.002
    answerLivingReward = 1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

### It is not possible to get 99% optimal
# policy with 50 iterations for this world, as 
# 50iter isnt enough for converge what epsilo 
# or learningrate we take - than its not possible.
def question6():
    answerEpsilon = 0.6
    answerLearningRate = 0.5
    return 'NOT POSSIBLE'
    # return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print 'Answers to analysis questions:'
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print '  Question %s:\t%s' % (q, str(response))
