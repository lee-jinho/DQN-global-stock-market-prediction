import numpy as np
import tensorflow as tf
import random 
import math 

# no need tensorflow
class exRep:

    def __init__( self, M, width, height ) :
        self.M          = M
        self.W          = width
        self.H          = height

        self.curS       = list()    # listof Matrix
        self.curA       = list()    # listof lenth 3 onehot vector
        self.curR       = list()    # listof Scalar
        self.nxtS       = list()    # listof Matrix

        # No Terminal State


    def remember ( self, curS, curA, curR, nxtS ):

        # remember current experience
        self.curS.append   ( curS )
        self.curA.append   ( curA )
        self.curR.append   ( round( curR, 4)  )
        self.nxtS.append    ( nxtS )

        # delete oldest experience
        if( len( self.curS ) > self.M ):
            del self.curS[0]
            del self.curA[0]
            del self.curR[0]
            del self.nxtS[0]


    def get_Batch   ( self, sessT, QA_Tuple, state_PH, isTrain_PH, Beta, numActions, Gamma ):

        curSs   = np.zeros( (Beta, self.H, self.W ) )
        curAs   = np.zeros( (Beta, numActions ) )
        Targets = np.zeros( (Beta, numActions ) )

        # get batchsize Beta random index from Memory Size List
        rIdxs   = random.sample( range( len( self.curS ) ), Beta )


        for k in range ( Beta ):

            input_kth   = self.curS[ rIdxs[k] ]
            action_kth  = self.curA[ rIdxs[k] ]

            QAValues    = sessT.run( QA_Tuple, feed_dict={ state_PH:self.nxtS[rIdxs[k]].reshape(1,self.H,self.W),isTrain_PH:False } )
            nxtQs       = QAValues[0]

            target_kth  = np.zeros( numActions)
            target_kth[ np.argmax(action_kth)] = self.curR[ rIdxs[k] ]  + Gamma * nxtQs[0][np.argmax(nxtQs[0])]

            curSs[k]    = input_kth
            curAs[k]    = action_kth
            Targets[k]  = target_kth
    
        return curSs, curAs, Targets
    
