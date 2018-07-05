import numpy as np
import os
import struct
import time

"""    Description:

    This is code that Mark wrote to open the multifile format
    in compressed mode, translated to python.
    This seems to work for DALSA, FCCD and EIGER in compressed mode.
    It should be included in the respective detector.i files
    Currently, this refers to the compression mode being '6'
    Each file is image descriptor files chunked together as follows:
            Header (1024 bytes)
    |--------------IMG N begin--------------|
    |                   Dlen
    |---------------------------------------|
    |       Pixel positions (dlen*4 bytes   |
    |      (0 based indexing in file)       |
    |---------------------------------------|
    |    Pixel data(dlen*bytes bytes)       |
    |    (bytes is found in header          |
    |    at position 116)                   |
    |--------------IMG N end----------------|
    |--------------IMG N+1 begin------------|
    |----------------etc.....---------------|


     Header contains 1024 bytes version name, 'beam_center_x', 'beam_center_y',
        'count_time', 'detector_distance', 'frame_time', 'incident_wavelength',
        'x_pixel_size', 'y_pixel_size', bytes per pixel (either 2 or 4
        (Default)), Nrows, Ncols, Rows_Begin, Rows_End, Cols_Begin, Cols_End,



"""


# TODO : split into RO and RW classes
class MultifileAPS:
    '''
    Re-write multifile from scratch.

    '''
    HEADER_SIZE = 1024

    def __init__(self, filename, mode='rb', nbytes=2):
        '''
            Prepare a file for reading or writing.
            mode : either 'rb' or 'wb'
            numimgs: num images
        '''
        if mode != 'rb' and mode != 'wb':
            raise ValueError("Error, mode must be 'rb' or 'wb'"
                             "got : {}".format(mode))
        self._filename = filename
        self._mode = mode

        self._nbytes = nbytes
        if nbytes == 2:
            self._dtype = '<i2'
        elif nbytes == 4:
            self._dtype = '<i4'

        # open the file descriptor
        # create a memmap
        if mode == 'rb':
            # self._fd = np.memmap(filename, dtype='c')
            self._fd = open(filename, "rb")
        elif mode == 'wb':
            self._fd = open(filename, "wb")
        # frame number currently on
        self.index()
        self.beg = 0
        self.end = self.Nframes-1

        # these are only necessary for writing
        hdr = self._read_header(0)
        self._rows = int(hdr['rows'])
        self._cols = int(hdr['cols'])

    def rdframe(self, n):
        # read header then image
        pos, vals = self._read_raw(n)
        img = np.zeros((self._rows*self._cols,))
        img[pos] = vals
        return img.reshape((self._rows, self._cols))

    def rdrawframe(self, n):
        # read header then image
        return self._read_raw(n)

    def index(self):
        ''' Index the file by reading all frame_indexes.
            For faster later access.
        '''
        print('Indexing file...')
        t1 = time.time()
        cur = 0
        file_bytes = len(self._fd)

        self.frame_indexes = list()
        while cur < file_bytes:
            self.frame_indexes.append(cur)
            # first get dlen, 4 bytes

            self._fd.seek(cur+152, os.SEEK_SET)
            # dlen = np.frombuffer(self._fd[cur+152:cur+156], dtype="<u4")[0]
            dlen = np.fromfile(self._fd, dtype=np.uint32, count=1)[0]
            print("found {} bytes".format(dlen))
            # self.nbytes is number of bytes per val
            cur += 1024 + dlen*(4+self._nbytes)
            # break

        self.Nframes = len(self.frame_indexes)
        t2 = time.time()
        print("Done. Took {} secs for {} frames".format(t2-t1, self.Nframes))

    def _read_header(self, n):
        ''' Read header from current seek position.'''
        if n > self.Nframes:
            raise KeyError("Error, only {} frames, asked for {}"
                           .format(self.Nframes, n))
        # read in bytes
        cur = self.frame_indexes[n]
        # header_raw = self._fd[cur:cur + self.HEADER_SIZE]
        header = dict()
        self._fd.seek(cur + 108, os.SEEK_SET)
        header['rows'] = np.fromfile(self._fd, dtype=self._dtype, count=1)[0]
        self._fd.seek(cur + 112, os.SEEK_SET)
        header['cols'] = np.fromfile(self._fd, dtype=self._dtype, count=1)[0]
        self._fd.seek(cur + 116, os.SEEK_SET)
        header['nbytes'] = np.fromfile(self._fd, dtype=self._dtype, count=1)[0]
        self._fd.seek(cur + 152, os.SEEK_SET)
        header['dlen'] = np.fromfile(self._fd, dtype=self._dtype, count=1)[0]

        self._dlen = header['dlen']
        self._nbytes = header['nbytes']

        return header

    def _read_raw(self, n):
        ''' Read from raw.
            Reads from current cursor in file.
        '''
        if n > self.Nframes:
            raise KeyError("Error, only {} frames, asked for {}"
                           .format(self.Nframes, n))
        cur = self.frame_indexes[n] + 1024
        dlen = self._read_header(n)['dlen']

        # pos = self._fd[cur: cur+dlen*4]
        self._fd.seek(cur, os.SEEK_SET)
        pos = np.fromfile(self._fd, dtype=np.uint32, count=dlen)
        cur += dlen*4
        # pos = np.frombuffer(pos, dtype='<i4')

        # TODO: 2-> nbytes
        vals = np.fromfile(self._fd, dtype=self._dtype, count=dlen)
        # vals = self._fd[cur: cur+dlen*2]
        # not necessary
        cur += dlen*2
        # vals = np.frombuffer(vals, dtype=self._dtype)

        return pos, vals

    def _write_header(self, dlen, rows, cols):
        ''' Write header at current position.'''
        self._rows = rows
        self._cols = cols
        self._dlen = dlen
        # byte array
        header = np.zeros(self.HEADER_SIZE, dtype="c")
        # write the header dlen
        header[152:156] = np.array([dlen], dtype="<i4").tobytes()
        # rows
        header[108:112] = np.array([rows], dtype="<i4").tobytes()
        # colds
        header[112:116] = np.array([cols], dtype="<i4").tobytes()
        self._fd.write(header)

    def write_raw(self, pos, vals):
        ''' Write a raw set of values for the next chunk.'''
        rows = self._rows
        cols = self._cols
        dlen = len(pos)
        self._write_header(dlen, rows, cols)
        # now write the pos and vals in series
        pos = pos.astype(self._dtype)
        vals = vals.astype(self._dtype)
        self._fd.write(pos)
        self._fd.write(vals)


# TODO : split into RO and RW classes
class MultifileBNL:
    '''
    Re-write multifile from scratch.

    '''
    HEADER_SIZE = 1024

    def __init__(self, filename, mode='rb', version=2):
        '''
            Prepare a file for reading or writing.
            mode : either 'rb' or 'wb'

            version : int, optional
                version 1 is old bnl format
                version 2 is the new format
        '''
        self._version = version
        if mode == 'wb':
            raise ValueError("Write mode 'wb' not supported yet")

        if mode != 'rb' and mode != 'wb':
            raise ValueError("Error, mode must be 'rb' or 'wb'"
                             "got : {}".format(mode))

        self._filename = filename
        self._mode = mode

        # open the file descriptor
        # create a memmap
        if mode == 'rb':
            # self._fd = np.memmap(filename, dtype='c')
            self._fd = open(filename, "rb")
        elif mode == 'wb':
            self._fd = open(filename, "wb")

        # these are only necessary for writing
        self.md = self._read_main_header()
        self._rows = int(self.md['nrows'])
        self._cols = int(self.md['ncols'])

        # some initialization stuff
        self.nbytes = self.md['bytes']
        if (self.nbytes == 2):
            self.valtype = np.uint16
        elif (self.nbytes == 4):
            self.valtype = np.uint32
        elif (self.nbytes == 8):
            self.valtype = np.float64

        # frame number currently on
        self.index()

    def __len__(self):
        return self.Nframes

    def index(self):
        ''' Index the file by reading all frame_indexes.
            For faster later access.
        '''
        print('Indexing file...')
        t1 = time.time()
        cur = self.HEADER_SIZE
        file_bytes = os.path.getsize(self._filename)
        # file_bytes = len(self._fd)

        self.frame_indexes = list()
        # for convenience
        self.frame_dlens = list()
        while cur < file_bytes:
            self.frame_indexes.append(cur)
            # first get dlen, 4 bytes
            #dlen = np.frombuffer(self._fd[cur:cur+4], dtype="<u4")[0]
            self._fd.seek(cur,os.SEEK_SET)
            #dlen = np.frombuffer(self._fd[cur+152:cur+156], dtype="<u4")[0]
            dlen = np.fromfile(self._fd, dtype=np.uint32,count=1)[0]
            self.frame_dlens.append(dlen)
            #print("found {} bytes".format(dlen))
            # self.nbytes is number of bytes per val
            cur += 4 + dlen*(4+self.nbytes)
            # break

        self.Nframes = len(self.frame_indexes)
        t2 = time.time()
        print("Done. Took {} secs for {} frames".format(t2-t1, self.Nframes))

    def _read_main_header(self):
        ''' Read header from current seek position.

            Extracting the header was written by Yugang Zhang. This is BNL's
            format.
            1024 byte header +
            4 byte dlen + (4 + nbytes)*dlen bytes
            etc...
            Format:
                unsigned int beam_center_x;
                unsigned int beam_center_y;
        '''
        # read in bytes
        # header is always from zero
        # header_raw = self._fd[cur:cur + self.HEADER_SIZE]
        ms_keys = ['beam_center_x', 'beam_center_y', 'count_time',
                   'detector_distance', 'frame_time', 'incident_wavelength',
                   'x_pixel_size', 'y_pixel_size', 'bytes', 'nrows', 'ncols',
                   'rows_begin', 'rows_end', 'cols_begin', 'cols_end']

        self._fd.seek(0, os.SEEK_SET)
        br = self._fd.read(1024)
        # magic = struct.unpack('@16s', br[:16])
        md_temp = struct.unpack('@8d7I916x', br[16:])
        self.md = dict(zip(ms_keys, md_temp))
        return self.md

    def _read_raw(self, n):
        ''' Read from raw.
            Reads from current cursor in file.
        '''
        if n > self.Nframes:
            raise KeyError("Error, only {} frames, asked for {}"
                           .format(self.Nframes, n))
        # dlen is 4 bytes
        cur = self.frame_indexes[n] + 4 # +4 for dlen

        dlen = self.frame_dlens[n]

        self._fd.seek(cur,os.SEEK_SET)

        pos = np.fromfile(self._fd, dtype=np.uint32,count=dlen)
        vals = np.fromfile(self._fd, dtype=self.valtype,count=dlen)

        return pos, vals

    def rdframe(self, n):
        # read header then image
        pos, vals = self._read_raw(n)
        img = np.zeros((self._rows*self._cols,))
        img[pos] = vals
        # trying to retain backwards compatibility of the old file
        if self._version > 1:
            img = img.reshape((self._rows, self._cols))
        else:
            img = img.reshape((self._cols, self._rows))
        return img

    def rdrawframe(self, n):
        # read header then image
        return self._read_raw(n)


class MultifileBNLCustom(MultifileBNL):
    def __init__(self, filename, beg=0, end=None, **kwargs):
        super().__init__(filename, **kwargs)
        self.beg = beg
        if end is None:
            end = self.Nframes-1
        self.end = end

    def rdframe(self, n):
        if n > self.end:
            raise IndexError("Index out of range")
        return super().rdframe(n - self.beg)

    def rdrawframe(self, n):
        return super().rdrawframe(n - self.beg)
