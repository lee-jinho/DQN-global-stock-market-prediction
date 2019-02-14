import DataPPRL as DRL
import train as TR



##############################################################################
# tensorflow  1.2.0
# Ubuntu 16.04
##############################################################################

FSize           = 5     
PSize           = 2
PStride         = 2
NumAction       = 3



# hyper parameters described in the paper 
#################################################################################
maxiter         = 5000000       # maxmimum iteration number         
learning_rate   = 0.00001       # learning rate
epsilon_min     = 0.1           # minimum epsilon

W               = 32            # input matrix size
M               = 1000          # memory buffer capacity
B               = 10            # parameter theta  update interval               
C               = 1000          # parameter theta^* update interval ( TargetQ )
Gamma           = 0.99          # discount factor
P               = 0             # transaction panalty while training.  0.05 (%) for training, 0 for testing
Beta            = 32            # batch size
#################################################################################

# initialize
DRead           = DRL.DataReaderRL()
Model           = TR.trainModel( 1.0, epsilon_min, maxiter, Beta, B , C, learning_rate, P  )



######## Test Model ###########
'''
# folder list for testing 
folderlist                          =  DRead.get_filelist(  '../Sample_Testing/')
sess,saver, state, isTrain, rho_eta = Model.TestModel_ConstructGraph( W,W,FSize,PSize,PStride,NumAction )

for i in range ( 0, len( folderlist) ):

    print folderlist[i]
   
    filepathX       =   folderlist[i] + 'inputX.txt'
    filepathY       =   folderlist[i] + 'inputY.txt' 

    XData           =   DRead.readRaw_generate_X( filepathX, W, W )
    YData           =   DRead.readRaw_generate_Y( filepathY, len(XData), len(XData[0]) )   

    Model.set_Data                          ( XData, YData )
    Model.Test_Neutralized_Portfolio        ( sess, saver, state, isTrain, rho_eta, W, W, NumAction )
    Model.Test_TopBottomK_Portfolio         ( sess, saver, state, isTrain, rho_eta, W, W, NumAction,  0.2 )
'''
###################################

########## Train Model ############


# folder path for training
filepathX       =   '../Sample_Training/WH32_32_2017_2018/inputX.txt'
filepathY       =   '../Sample_Training/WH32_32_2017_2018/inputY.txt'

XData           = DRead.readRaw_generate_X( filepathX, W, W )                       # input chart
YData           = DRead.readRaw_generate_Y( filepathY, len(XData), len(XData[0]) )  # L_c^t  
Model.set_Data      ( XData, YData)
Model.trainModel    ( W,W, FSize, PSize, PStride, NumAction, M, Gamma )

####################################




