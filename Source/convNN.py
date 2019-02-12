import tensorflow as tf
import numpy as np
import math


class ConstructCNN:

    def __init__        ( self, Height, Width, FSize, PSize, PStride, NumAction ):

        self.H          = Height
        self.W          = Width
        self.FSize      = FSize
        self.PSize      = PSize
        self.PStride    = PStride
        self.NumAction  = NumAction
    


    def QValue   ( self, state, isTrain ):


        X           = tf.reshape            ( state, [ -1, self.H, self.W, 1] )
        M,V         = tf.nn.moments         ( X, [0,1,2,3] )
        X           = self.normalize_input  ( X, M, V )

        # CNN Layer
        Layer1, M1,V1    = self.stackedLayer ( 'L1', X,      self.FSize, self.PSize, self.PStride,  1,   16,  2, isTrain )
        Layer2, M2,V2    = self.stackedLayer ( 'L2', Layer1, self.FSize, self.PSize, self.PStride,  16,  32,  2, isTrain )
 
        # Fully Connected Network
        # L6    :  Batch x inputSize
        L6          = tf.contrib.layers.flatten( Layer2 )
        FC1         = self.FCLayer      ( "FC_1", L6, int( L6.get_shape()[-1]), 32, isTrain )
        FC2         = self.FinalLayer   ( "FC_2", FC1, 32, self.NumAction, isTrain )

        # FC2 : Batch * 3
        rho         = FC2
        eta         = tf.one_hot(  tf.argmax( rho, 1 ), self.NumAction, on_value = 1, off_value = 0, dtype = tf.int32 )

        return rho, eta 


    def optimize_Q          ( self, Q, A, Y, batchsize, learning_rate ):

        # Q : Batch * numaction
        # A : Batch * numaction 

        # update BatchNorm var first, and then update Loss, Opt
        updates     = tf.get_collection ( tf.GraphKeys.UPDATE_OPS )
        trablevars  = tf.trainable_variables()

        with tf.control_dependencies ( updates ):

            Loss    = tf.reduce_sum             ( tf.square(  Y - (Q*A) ) ) /   batchsize 
            opt     = tf.train.AdamOptimizer    ( learning_rate )

            grads   = opt.compute_gradients     ( Loss )
            minz    = opt.minimize              ( Loss )
        
        return  Loss, grads, updates, trablevars, minz 


    def FinalLayer             ( self, Name, Lin, inputSize, LayerSize, isTrain ):

        # inputSize, LayerSize
        W   = tf.get_variable( Name, [ inputSize, LayerSize],  initializer = tf.contrib.layers.xavier_initializer()  )
        B   = tf.get_variable( Name + "_B" ,  initializer = tf.truncated_normal( [1,LayerSize],stddev = 0.01) )
        Out = tf.matmul (Lin, W) + B
        return Out


    def FCLayer             ( self, Name, Lin, inputSize, LayerSize, isTrain ):

        # inputSize, LayerSize
        W   = tf.get_variable( Name, [ inputSize, LayerSize],  initializer = tf.contrib.layers.xavier_initializer()  )
        B   = tf.get_variable( Name + "_B" ,  initializer = tf.truncated_normal( [1,LayerSize],stddev = 0.01) )

        Out = tf.matmul (Lin, W) + B
        BN  = tf.contrib.layers.batch_norm( Out, scale = True, is_training = isTrain, scope = Name )
        return tf.nn.relu( BN )


    def stackedLayer   ( self, Name, Lin, Fsize, poolsize,  poolstride,  inSize, outSize,  numLayer, isTrain ):

        L       = self.convLayer                (   Name+'_0' , Lin, Fsize, inSize,  outSize )
        BN      = tf.contrib.layers.batch_norm  (   L, scale = True, 
                                                    is_training=isTrain, scope=Name+'_0' )
        A       = tf.nn.relu                    (   BN )

        for i in range ( 1, numLayer ):
            L   = self.convLayer                (   Name+ '_' +str(i), A, Fsize, outSize, outSize )
            BN  = tf.contrib.layers.batch_norm  (   L, scale = True, 
                                                    is_training = isTrain, scope=Name+'_'+str(i) )

            A   = tf.nn.relu                    (   BN )
        
        Mlast, Vlast    = tf.nn.moments( BN, [ 0,1,2,3] )  
        Lout            = tf.nn.max_pool    (  A,   [1,poolsize,poolsize,1], [1,poolstride,poolstride,1], 'VALID' )
        return Lout, Mlast, Vlast


    def convLayer           ( self, Name, Lin, Fsize,Channel, Osize ):

        # BHWC        
        W   = tf.get_variable   ( Name, [Fsize,Fsize,Channel,Osize], initializer = tf.contrib.layers.xavier_initializer() )
        L   = tf.nn.conv2d      ( Lin, W,   [1,1,1,1], 'SAME' )
        return L


    def normalize_input     ( self, X, M, V ):
        return ( X - M ) / tf.sqrt ( V )
        

