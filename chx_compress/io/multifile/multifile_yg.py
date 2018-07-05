'''
Multifile by Yugang Zhang (yuzhang@bnl.gov)

'''
import numpy as np
import struct
import os

class Multifile:
    '''The class representing the multifile.
        The recno is in 1 based numbering scheme (first record is 1)
	This is efficient for reading in increasing order.
	Note: reading same image twice in a row is like reading an earlier
	numbered image and means the program starts for the beginning again.

    '''
    def __init__(self,filename,beg,end, reverse=False):
        '''Multifile initialization. Open the file.
            Here I use the read routine which returns byte objects
            (everything is an object in python). I use struct.unpack
            to convert the byte object to other data type (int object
            etc)
            NOTE: At each record n, the file cursor points to record n+1
        '''
        self.FID = open(filename,"rb")
#        self.FID.seek(0,os.SEEK_SET)
        self.filename = filename
        #br: bytes read
        br = self.FID.read(1024)
        self.beg=beg
        self.end=end
        ms_keys = ['beam_center_x', 'beam_center_y', 'count_time', 'detector_distance',
           'frame_time', 'incident_wavelength', 'x_pixel_size', 'y_pixel_size',
           'bytes',
            'nrows', 'ncols', 'rows_begin', 'rows_end', 'cols_begin', 'cols_end'
          ]

        magic = struct.unpack('@16s', br[:16])
        md_temp =  struct.unpack('@8d7I916x', br[16:])
        self.md = dict(zip(ms_keys, md_temp))
        if reverse:
            nrows  = self.md['nrows'] 
            ncols  = self.md['ncols']
            self.md['nrows'] = ncols
            self.md['ncols'] = nrows
            rbeg = self.md['rows_begin']
            rend = self.md['rows_end']
            cbeg = self.md['cols_begin']
            cend = self.md['cols_end']
            self.md['rows_begin']=cbeg
            self.md['rows_end']=cend
            self.md['cols_begin']=rbeg
            self.md['cols_end']=rend

        self.imgread=0
        self.recno = 0
        # some initialization stuff
        self.byts = self.md['bytes']
        if (self.byts==2):
            self.valtype = np.uint16
        elif (self.byts == 4):
            self.valtype = np.uint32
        elif (self.byts == 8):
            self.valtype = np.float64
        #now convert pieces of these bytes to our data
        self.dlen =np.fromfile(self.FID,dtype=np.int32,count=1)[0]

        # now read first image
        #print "Opened file. Bytes per data is {0img.shape = (self.rows,self.cols)}".format(self.byts)

    def _readHeader(self):
        self.dlen =np.fromfile(self.FID,dtype=np.int32,count=1)[0]

    def _readImageRaw(self):

        p= np.fromfile(self.FID, dtype = np.int32,count= self.dlen)
        v= np.fromfile(self.FID, dtype = self.valtype,count= self.dlen)
        self.imgread=1
        return(p,v)

    def _readImage(self):
        (p,v)=self._readImageRaw()
        img = np.zeros( (  self.md['ncols'], self.md['nrows']   ) )
        np.put( np.ravel(img), p, v )
        return(img)

    def seekimg(self,n=None):

        '''Position file to read the nth image.
            For now only reads first image ignores n
        '''
        # the logic involving finding the cursor position
        if (n is None):
            n = self.recno
        if (n < self.beg or n > self.end):
            raise IndexError('Error, record out of range')
        #print (n, self.recno, self.FID.tell() )
        if ((n == self.recno)  and (self.imgread==0)):
            pass # do nothing

        else:
            if (n <= self.recno): #ensure cursor less than search pos
                self.FID.seek(1024,os.SEEK_SET)
                self.dlen =np.fromfile(self.FID,dtype=np.int32,count=1)[0]
                self.recno = 0
                self.imgread=0
                if n == 0:
                    return
            #have to iterate on seeking since dlen varies
            #remember for rec recno, cursor is always at recno+1
            if(self.imgread==0 ): #move to next header if need to
                self.FID.seek(self.dlen*(4+self.byts),os.SEEK_CUR)
            for i in range(self.recno+1,n):
                #the less seeks performed the faster
                #print (i)
                self.dlen =np.fromfile(self.FID,dtype=np.int32,count=1)[0]
                #print 's',self.dlen
                self.FID.seek(self.dlen*(4+self.byts),os.SEEK_CUR)

            # we are now at recno in file, read the header and data
            #self._clearImage()
            self._readHeader()
            self.imgread=0
            self.recno = n
    def rdframe(self,n):
        if self.seekimg(n)!=-1:
            return(self._readImage())

    def rdrawframe(self,n):
        if self.seekimg(n)!=-1:
             return(self._readImageRaw())
