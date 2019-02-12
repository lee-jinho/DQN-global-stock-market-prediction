import tensorflow as tf
import numpy as np
import random
import convNN as CNN
import exReplay as exR


gpu_config = tf.ConfigProto()  
gpu_config.gpu_options.allow_growth = True # only use required resource(memory)
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.5 # restrict to 50%


class trainModel:

    def __init__   ( self,  epsilon_init, epsilon_min, maxiter, Beta, B,C,  learning_rate, P  ):

        self.DataX          = list() 
        self.DataY          = list()

        self.epsilon        = epsilon_init 
        self.epsilon_min    = epsilon_min

        self.maxiter        = maxiter
        self.Beta           = Beta
        self.learning_rate  = learning_rate 
        self.P              = P 
        self.B              = B
        self.C              = C

    def set_Data    ( self, DataX, DataY ):
        self.DataX = DataX
        self.DataY = DataY

        print 'X Data:  Comp#, Days# ', len( self.DataX ), len( self.DataX[0] )
        print 'Y Data:  Comp#, Days# ', len( self.DataY ), len( self.DataY[0] )

    def trainModel ( self, H,W, FSize, PSize, PStride, NumAction, M, Gamma  ):

        # place holder
        state       = tf.placeholder ( tf.float32, [None,H,W] )
        isTrain     = tf.placeholder ( tf.bool, [] )

        Action      = tf.placeholder ( tf.float32,  [ None,NumAction ] )
        Target      = tf.placeholder ( tf.float32,[ None,NumAction ] )

        # construct Graph
        C           = CNN.ConstructCNN( H,W, FSize, PSize, PStride, NumAction )
        rho_eta     = C.QValue    ( state, isTrain  )
        Loss_Tuple  = C.optimize_Q( rho_eta[0], Action, Target, self.Beta, self.learning_rate )

        sess        = tf.Session ( config = gpu_config )    # maintains network parameter theta
        sessT       = tf.Session ( config = gpu_config )    # maintains target networ parameter theta^*
        sess.run ( tf.global_variables_initializer () )

        # saver
        saver       = tf.train.Saver( max_to_keep = 20 )

        # copy inital
        saver.save      ( sess, 'DeepQ' )
        saver.restore   ( sess, 'DeepQ' )
        saver.restore   ( sessT, 'DeepQ' )

        # current experience
        preS    = np.empty( (1,H,W), dtype = np.float32 )
        preA    = np.empty( ( NumAction ), dtype = np.int32 )

        curS    = np.empty( (1,H,W), dtype = np.float32 )
        curA    = np.empty( (NumAction), dtype = np.int32 )
        curR    = 0
        nxtS    = np.empty( (H,W), dtype = np.float32 )

        memory  = exR.exRep( M, W, H )  # memory buffer
        b       = 1                     # iteration counter

        while True:

            #1.0 get random valid index c, t
            c       = random.randrange( 0, len( self.DataX ) )
            t       = random.randrange( 1, len( self.DataX[c] ) -1  )

            #1.1 get preS
            preS    = self.DataX[c][t-1]
            
            #1.2 get preA by applying epsilon greedy policy to preS
            if( self.randf(0,1) <= self.epsilon):
                preA        = self.get_randaction   ( NumAction ) 
            else:                    
                QAValues    = sess.run              ( rho_eta, feed_dict={ state: preS.reshape(1,H,W), isTrain:False } )
                preA        = QAValues[1].reshape   ( NumAction )

            #1.3 get curS
            curS    = self.DataX[c][t]

            #1.4 get curA by applying epsilon greedy policy to curS
            if( self.randf(0,1) <= self.epsilon):
                curA        = self.get_randaction   ( NumAction ) 
            else:                    
                QAValues    = sess.run              ( rho_eta, feed_dict={ state: curS.reshape(1,H,W), isTrain:False } )
                curA        = QAValues[1].reshape   ( NumAction )

            #1.5 get current reward and next state
            curR    = self.get_reward( preA, curA, self.DataY[c][t], self.P )
            nxtS    = self.DataX[c][t+1]

            #1.6 remember experience : tuple of curS, curA, curR, nxtS   
            memory.remember( curS, curA, curR, nxtS )

            #1.7: set epsilon                       
            if ( self.epsilon > self.epsilon_min ):
                self.epsilon = self.epsilon * 0.999999  

            #2: update network parameter theta  every  B iteration
            if ( len( memory.curS ) >= M ) and( b % self.B == 0 ) :

                #2.1:  update Target network parameter theta^*
                if( b % ( self.C * self.B ) == 0 )  : 
                    saver.save      ( sess, 'DeepQ'  )
                    saver.restore   ( sessT, 'DeepQ' )

                #2.2: sample Beta size batch from memory buffer and take gradient step with repect to network parameter theta 
                S,A,Y   = memory.get_Batch  ( sessT, rho_eta, state, isTrain,  self.Beta, NumAction, Gamma )
                Opts    = sess.run          ( Loss_Tuple, feed_dict = { state:S, isTrain:True, Action:A, Target:Y }  )

                #2.3: print Loss 
                if( b % ( 100 * self.B  ) == 0 ):
                    print 'Loss: ' ,b, Opts[0] 

            #3: update iteration counter
            b   = b + 1

            #4: save model 
            if( b >= self.maxiter ):
                saver.save( sess, 'DeepQ' )
                print 'Finish! '
                return 0


    def validate_Neutralized_Portfolio       ( self, DataX, DataY, sess, rho_eta, state, isTrain, NumAction, H,W  ):
       
        # list
        N           = len( DataX )
        Days        = len( DataX[0] )
        curA        = np.zeros(( N, NumAction ))

        # alpha
        preAlpha_n  = np.zeros( N )
        curAlpha_n  = np.zeros( N )
        posChange   = 0

        # reward
        curR        = np.zeros( N )
        avgDailyR   = np.zeros( Days )


        # cumulative asset:  initialize cumAsset to 1.0
        cumAsset    = 1

        for t in range ( Days - 1 ):
    
            for c in range ( N ):
           
                #1: choose action from current state 
                curS        = DataX[c][t]
                QAValues    = sess.run  ( rho_eta, feed_dict={ state: curS.reshape(1,H,W), isTrain:False } )
                curA[c]     = np.round  ( QAValues[1].reshape( ( NumAction) ) )
            
            # set Neutralized portfolio for day t
            curAlpha_n  = self.get_NeutralizedPortfolio ( curA,  N  )

            for c in range ( N ) :

                #1: get daily reward sum 
                curR[c]                     = np.round(  curAlpha_n[c] * DataY[c][t], 8)
                avgDailyR[t]                = np.round(  avgDailyR[t] + curR[c], 8 )

                #2: pos change sum
                posChange                   = np.round(  posChange +  abs( curAlpha_n[c] - preAlpha_n[c] ), 8)
                preAlpha_n[c]               = curAlpha_n[c]


        # calculate cumulative return
        for t in range( Days ):
            cumAsset = round ( cumAsset + ( cumAsset * avgDailyR[t] * 0.01  ), 8 )

        print 'cumAsset ',  cumAsset
        return N, posChange, cumAsset


    def validate_TopBottomK_Portfolio       ( self, DataX, DataY, sess, rho_eta, state, isTrain, NumAction, H,W, K  ):

        # list
        N           = len( DataX )
        Days        = len( DataX[0] )

        # alpha
        preAlpha_s  = np.zeros( N )
        curAlpha_s  = np.zeros( N )
        posChange   = 0

        # reward
        curR        = np.zeros( N )
        avgDailyR   = np.zeros( Days )
      
        # cumulative asset: initialize curAsset to 1.0
        cumAsset    = 1

        # action value for Signals and Threshold for Top/Bottom K 
        curActValue = np.zeros( (N, NumAction ) )
        LongSignals = np.zeros( N )

        UprTH       = 0
        LwrTH       = 0

        for t in range ( Days - 1 ):

            for c in range ( N ):
           
                #1: choose action from current state 
                curS            = DataX[c][t]
                QAValues        = sess.run  ( rho_eta, feed_dict={ state: curS.reshape(1,H,W), isTrain:False } )
                curActValue[c]  = np.round  ( QAValues[0].reshape( ( NumAction) ), 4 )
                LongSignals[c]  = curActValue[c][0] - curActValue[c][2]

            # set Top/Bottom portfolio for day t
            UprTH, LwrTH        = self.givenLongSignals_getKTH  ( LongSignals, K, t  ) 
            curAlpha_s          = self.get_TopBottomPortfolio   ( UprTH, LwrTH, LongSignals, N )       

            for c in range ( N ):

                #1: get daily reward sum
                curR[c]                     = np.round(  curAlpha_s[c] * DataY[c][t], 8)
                avgDailyR[t]                = np.round(  avgDailyR[t] + curR[c], 8 )

                #2: pos change sum
                posChange                   = np.round(  posChange +  abs( curAlpha_s[c] - preAlpha_s[c] ), 8)
                preAlpha_s[c]               = curAlpha_s[c]


        # calculate cumulative return
        for t in range( Days ):
            cumAsset = round (cumAsset + ( cumAsset * avgDailyR[t] * 0.01  ), 8 )

        print 'cumAsset ',  cumAsset
        return N, posChange, cumAsset


    def TestModel_ConstructGraph    ( self, H,W, FSize, PSize, PStride,  NumAction  ):

        # place holder
        state       = tf.placeholder ( tf.float32, [None,H,W] )
        isTrain     = tf.placeholder ( tf.bool, [] )

        #print tf.shape( isTrain)
        #print(tf.__version__)

        # construct Graph
        C           = CNN.ConstructCNN( H,W, FSize, PSize, PStride, NumAction )
        rho_eta     = C.QValue    ( state, isTrain  )

        sess        = tf.Session ( config = gpu_config )
        saver       = tf.train.Saver()

        return sess, saver, state, isTrain, rho_eta

    def Test_TopBottomK_Portfolio   ( self, sess, saver, state, isTrain, rho_eta,  H,W, NumAction, TopK  ):

        saver.restore( sess, 'DeepQ' )
        Outcome     = self.validate_TopBottomK_Portfolio (  self.DataX, self.DataY, sess, rho_eta, state, isTrain, NumAction, H,W, TopK  )

        print 'NumComp#: ',  Outcome[0],  'Transactions: ', Outcome[1]/2, 'cumulative asset',Outcome[2] 
        self.writeResult_daily( 'TestResult.txt', Outcome,  len ( self.DataX[0] ) -1  )


    def Test_Neutralized_Portfolio  ( self, sess, saver, state, isTrain,  rho_eta,  H,W, NumAction  ):

        saver.restore( sess, 'DeepQ' )
        Outcome     = self.validate_Neutralized_Portfolio (  self.DataX, self.DataY, sess, rho_eta, state, isTrain, NumAction, H,W  )

        print 'NumComp#: ',  Outcome[0],  'Transactions: ', Outcome[1]/2, 'cumulative asset',Outcome[2] 
        self.writeResult_daily ( 'TestResult.txt', Outcome, len( self.DataX[0] ) -1  )

 
    def get_NeutralizedPortfolio         ( self, curA, N ):         
        
        alpha       = np.zeros( N )
        avg         = 0
        
        # get average
        for c in range ( N ):
            alpha[c]    = 1 - np.argmax( curA[c] )
            avg         = avg + alpha[c]
            
        avg     = np.round( avg / N, 4 )

        #set alpha
        sum_a       = 0
        for c in range ( N ):
            alpha[c]= np.round( alpha[c] - avg, 4 )
            sum_a   = np.round( sum_a + abs(alpha[c]), 4 )

        # set alpha
        if sum_a == 0 :
            return alpha

        for c in range ( N ):
            alpha[c] =np.round(  alpha[c] / sum_a, 8 )

        return alpha


    def givenLongSignals_getKTH       ( self, LongSignals, K, t  ):
        
        Num         =  int( len(LongSignals) * K)
        SortedLongS =  np.sort( LongSignals )

        return SortedLongS[len(LongSignals) - Num], SortedLongS[Num-1]


    def get_TopBottomPortfolio              ( self, UprTH, LwrTH, LongSignals, N ):

        alpha   = np.zeros( N )
        sum_a   = 0

        for c in range ( N ):
            if LongSignals[c] >= UprTH:
                alpha[c] = 1
                sum_a = sum_a + 1
            elif LongSignals[c] <= LwrTH:
                alpha[c] = -1
                sum_a = sum_a+1
            else:
                alpha[c] = 0

        if sum_a == 0: 
            return alpha

        for c in range ( N ) :
            alpha[c] = np.round( alpha[c] / float(sum_a), 8 )

        return alpha
        

    def randf           ( self,  s, e):
        return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s;

    def get_randaction  ( self,  numofaction ) :
        actvec      =  np.zeros( (numofaction), dtype = np.int32 )
        idx         =  random.randrange(0,numofaction)
        actvec[idx] = 1
        return actvec


    def get_reward    ( self, preA, curA, inputY, P ):
        
        # 1,0,-1 is assined to pre_act, cur_act 
        # for action long, neutral, short respectively
        pre_act = 1- np.argmax( preA ) 
        cur_act = 1- np.argmax( curA ) 

        return  (cur_act * inputY) - P * abs( cur_act - pre_act ) 


    def writeResult_daily    ( self,  filename,  outcome, numDays ):
        f = open( filename, 'a' )

        f.write( 'Comp#,'       + str( outcome[0]) + ',' )
        f.write( 'Days#'        + str( numDays-1 ) + ',' )
        f.write( 'TR#,'         + str( round( outcome[1]/2, 4) ) + ',' )
        f.write( 'FinalAsset,'  + str( round( outcome[2], 4 )) )
        
        f.write("\n")
        f.close()


