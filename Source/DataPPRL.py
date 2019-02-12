import numpy as np
import os


class DataReaderRL:

    def __init__ ( self ):
        self.a = 0


    def get_filelist ( self, rootpath ):

        pathlist = list()

        country = os.listdir( rootpath )
        for i in range(  len(country) ):
            country[i] = rootpath + str( country[i] ) + '/'

        datelist = list()
        for i in range( len(country) ):
            datelist = os.listdir( country[i] ) 

            for j in range( len (datelist) ):
                pathlist.append( country[i] + datelist[j] + '/')


        pathlist.sort()

        #for i in range( len(pathlist) ):
         #   print pathlist[i]
        print 'numof all data : ', len( pathlist )
        return pathlist

 
    def readRaw_generate_X  (self, filepath, height, width ):

        # Generate height by wdith   input chart image
        

        f       =   open ( filepath, 'r' )
        rawdata =   f.read()
        rawdata =   rawdata.split( '\nF\n'  )
        DataX   =   list()
        N       =   len( rawdata ) - 1
        Days    =   len( rawdata[0].split( '\nE\n' ) )

        for c in range( N ) :
            state_seq  = rawdata[c].split( '\nE\n' )

            # matrix seq for company c
            matrix_seq = list()
            for t in range ( Days ):
                matrix  = np.zeros( ( height, width ) )
                rows    = state_seq[t].split('\n')
                
                # input matrix on day t
                for r in range ( height ):
                    row  = rows[r].split( ' ' )
                    for w in range( width ):
                        matrix[r][w] = int( row[w] )

                matrix_seq.append( matrix )

            DataX.append ( matrix_seq )


        return DataX 
                                          
    def readRaw_generate_Y    ( self, filepath, N, Days ):

        # Generate input price change L_c^t

        f       =   open ( filepath, 'r' )
        rawdata =   f.read()
        rawdata =   rawdata.split( '\n'  )
        DataY   = list()

        if ( len(rawdata)-1) != (N*Days) :
            print 'number of input data is invalid'

        cnt     = 0
        for c in range ( N ) :
            return_seq = list()

            for t in range ( Days)  :
                return_seq.append( float( rawdata [cnt] ) )
                cnt = cnt + 1

            DataY.append ( return_seq )

 
        return DataY 






