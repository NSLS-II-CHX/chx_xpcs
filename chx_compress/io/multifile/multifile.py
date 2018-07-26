import os.path
import numpy as np
import os
import struct
import time
"""    Description:
     This is code that Mark wrote to open the multifile format
    in compressed mode, translated to python.
    This seems to work for DALSA and FCCD in compressed mode
    should be included in the respective detector.i files
    Currently, this refers to the compression mode being '6'
    Each file is image descriptor files chunked together as follows:

    |--------------IMG N begin--------------|
    |        Header (1024 bytes)            |
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

    Here is the file layout as follows:
    0: mode (4 bytes)
    4: compression (4 bytes)
    8: date (32 bytes)
    40: prefix (16 bytes)
    56: number (4 bytes)
    60: suffix (16 bytes)
    76: monitor (4 bytes)
    80: shutter (4 bytes)
    84: row_beg (4 bytes)
    88: row_end (4 bytes)
    92: col_beg (4 bytes)
    96: col_end (4 bytes)
    100: row_bin (4 bytes)
    104: col_bin (4 bytes)
    108: rows (4 bytes)
    112: cols (4 bytes)
    116: bytes (4 bytes)
    120: kinetics (4 bytes)
    124: kinwinsize (4 bytes)
    128: elapsed (8 bytes)
    136: preset (8 bytes) in seconds
    144: topup (4 bytes)
    148: inject (4 bytes)
    152: dlen (4 bytes)
    156: roi_number (4 bytes)
    160: buffer_number (4 bytes)
    164: systick (4 bytes) in clock cycles
    608: imageserver (4 bytes)
    612: CPUspeed (4 bytes)
    616: immversion (4 bytes)
    620: corecotick (4 bytes) in microseconds
    624: cameratype (4 bytes)
    628: threshold (4 bytes)
"""

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
            self._fd = open(filename, "rb")
            # frame number currently on
            self.index()
            self.beg = 0
            self.end = self.Nframes-1
            # these are only necessary for writing
            hdr = self._read_header(0)
            self._rows = int(hdr['rows'])
            self._cols = int(hdr['cols'])
        elif mode == 'wb':
            self._fd = open(filename, "wb")

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
        header_raw = self._fd[cur:cur + self.HEADER_SIZE]
        header = dict()
        # header['rows'] = np.frombuffer(header_raw[108:112],
        #   dtype=self._dtype)[0]
        # header['cols'] = np.frombuffer(header_raw[112:116],
        #   dtype=self._dtype)[0]
        # header['nbytes'] = np.frombuffer(header_raw[116:120],
        #   dtype=self._dtype)[0]
        # header['dlen'] = np.frombuffer(header_raw[152:156],
        #   dtype=self._dtype)[0]
        # print("dlen: {}\trows: {}\tcols: {}\tnbytes: {}\n"\
        #   .format(header['dlen'], header['rows'], header['cols'],
        #   header['nbytes']))
        # 0: mode (4 bytes)
        header['mode'] = np.frombuffer(header_raw[0:4], dtype=np.uint32)[0]
        # 4: compression (4 bytes)
        header['date'] = header_raw[8:8+16].tobytes()
        # 8: date (32 bytes)
        header['prefix'] = header_raw[40:40+16].tobytes()
        # 40: prefix (16 bytes)
        header['number'] = np.frombuffer(header_raw[56:56+4],
                                         dtype=np.uint32)[0]
        # 56: number (4 bytes)
        header['suffix'] = header_raw[60:60+16].tobytes()
        # 60: suffix (16 bytes)
        header['monitor'] = np.frombuffer(header_raw[76:76+4],
                                          dtype=np.uint32)[0]
        # 76: monitor (4 bytes)
        header['shutter'] = np.frombuffer(header_raw[80:80+4],
                                          dtype=np.uint32)[0]
        # 80: shutter (4 bytes)
        header['row_beg'] = np.frombuffer(header_raw[84:84+4],
                                          dtype=np.uint32)[0]
        # 84: row_beg (4 bytes)
        header['row_end'] = np.frombuffer(header_raw[88:88+4],
                                          dtype=np.uint32)[0]
        # 88: row_end (4 bytes)
        header['col_beg'] = np.frombuffer(header_raw[92:92+4],
                                          dtype=np.uint32)[0]
        # 92: col_beg (4 bytes)
        header['col_end'] = np.frombuffer(header_raw[96:96+4],
                                          dtype=np.uint32)[0]
        # 96: col_end (4 bytes)
        header['row_bin'] = np.frombuffer(header_raw[100:100+4],
                                          dtype=np.uint32)[0]
        # 100: row_bin (4 bytes)
        header['col_bin'] = np.frombuffer(header_raw[104:104+4],
                                          dtype=np.uint32)[0]
        # 104: col_bin (4 bytes)
        header['rows'] = np.frombuffer(header_raw[108:108+4],
                                       dtype=np.uint32)[0]
        # 108: rows (4 bytes)
        header['cols'] = np.frombuffer(header_raw[112:112+4],
                                       dtype=np.uint32)[0]
        # 112: cols (4 bytes)
        header['nbytes'] = np.frombuffer(header_raw[116:116+4],
                                         dtype=np.uint32)[0]
        # 116: bytes (4 bytes)
        header['kinetics'] = np.frombuffer(header_raw[120:120+4],
                                           dtype=np.uint32)[0]
        # 120: kinetics (4 bytes)
        header['kinwinsize'] = np.frombuffer(header_raw[124:124+4],
                                             dtype=np.uint32)[0]
        # 124: kinwinsize (4 bytes)
        header['elapsed'] = np.frombuffer(header_raw[128:128+4],
                                          dtype=np.uint32)[0]
        # 128: elapsed (8 bytes)
        header['preset'] = np.frombuffer(header_raw[136:136+8],
                                         dtype=np.float64)[0]
        # 136: preset (8 bytes) in seconds
        header['topup'] = np.frombuffer(header_raw[144:144+4],
                                        dtype=np.uint32)[0]
        # 144: topup (4 bytes)
        header['inject'] = np.frombuffer(header_raw[148:148+4],
                                         dtype=np.uint32)[0]
        # 148: inject (4 bytes)
        header['dlen'] = np.frombuffer(header_raw[152:152+4],
                                       dtype=np.uint32)[0]
        # 152: dlen (4 bytes)
        header['roi_number'] = np.frombuffer(header_raw[156:156+4],
                                             dtype=np.uint32)[0]
        # 156: roi_number (4 bytes)
        header['buffer_number'] = np.frombuffer(header_raw[160:160+4],
                                                dtype=np.uint32)[0]
        # 160: buffer_number (4 bytes)
        header['systick'] = np.frombuffer(header_raw[164:164+4],
                                          dtype=np.uint32)[0]
        # 164: systick (4 bytes) in clock cycles
        header['imageserver'] = np.frombuffer(header_raw[608:608+4],
                                              dtype=np.uint32)[0]
        # 608: imageserver (4 bytes)
        header['CPUspeed'] = np.frombuffer(header_raw[612:612+4],
                                           dtype=np.uint32)[0]
        # 612: CPUspeed (4 bytes)
        header['immversion'] = np.frombuffer(header_raw[616:616+4],
                                             dtype=np.uint32)[0]
        # 616: immversion (4 bytes)
        header['corecotick'] = np.frombuffer(header_raw[620:620+4],
                                             dtype=np.uint32)[0]
        # 620: corecotick (4 bytes) in microseconds
        header['cameratype'] = np.frombuffer(header_raw[624:624+4],
                                             dtype=np.uint32)[0]
        # 624: cameratype (4 bytes)
        header['threshold'] = np.frombuffer(header_raw[628:628+4],
                                            dtype=np.uint32)[0]
        # 628: threshold (4 bytes)

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

    def close(self):
        self._fd.close()


# TODO : split into RO and RW classes
class MultifileBNL:
    '''
    Re-write multifile from scratch.

    '''
    HEADER_SIZE = 1024

    def __init__(self, filename, mode='rb', version=2, md={}):
        '''
            Prepare a file for reading or writing.

            Parameters
            ----------
            filename : string
                the filename to read/write from
            mode: 'rb' or 'wb'
                the read/write mode
            version : int, optional
                version 1 is old bnl format
                version 2 is the new format
            md: dict, optional
                when writing, this needs to be set
                beam_center_x: (float) beam center x pos
                beam_center_y: (float) beam center y pos
                count_time: (float) exposure time
                detector_distance: (float) sample det distance
                frame_time: (float) frame time
                incident_wavelength: (float) wavelength
                x_pixel_size: (float) pixel x dimensions
                y_pixel_size: (float) pixel y dimensions
                bytes: (int) number of bytes per val
                nrows: (int) number of rows in an image
                ncols: (int) number of cols in an image
                rows_begin: beginning row number
                rows_end: end row number
                cols_begin: beginning col number
                cols_end: end col number
        '''
        self.md = md
        self._version = version
        # if mode == 'wb':
        #   raise ValueError("Write mode 'wb' not supported yet")

        if mode != 'rb' and mode != 'wb':
            raise ValueError("Error, mode must be 'rb' or 'wb'"
                             "got : {}".format(mode))

        self._filename = filename
        self._mode = mode

        # open the file descriptor
        # create a memmap
        if mode == 'rb':
            self._fd = open(filename, "rb")
            # these are only necessary for reading
            self.md = self._read_main_header()
            self._rows = int(self.md['nrows'])
            self._cols = int(self.md['ncols'])
        elif mode == 'wb':
            if os.path.exists(filename):
                msg = f"Warning: {filename} exists. Overwrite? (y/N)"
                result = input(msg)
                if result.lower() != "y":
                    raise ValueError("Error cannot continue")

            self._fd = open(filename, "wb")
            print("initializing for write, writing main header")
            self._write_main_header(self.md)

        # some initialization stuff
        self.nbytes = self.md['bytes']
        if (self.nbytes == 2):
            self.valtype = np.uint16
        elif (self.nbytes == 4):
            self.valtype = np.uint32
        elif (self.nbytes == 8):
            self.valtype = np.float64

        if mode == 'rb':
            # frame number currently on
            self.index()

    def close(self):
        self._fd.close()

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
        while cur < file_bytes:
            self.frame_indexes.append(cur)
            # first get dlen, 4 bytes

            # dlen = np.frombuffer(self._fd[cur:cur+4], dtype="<u4")[0]
            self._fd.seek(cur, os.SEEK_SET)
            # dlen = np.frombuffer(self._fd[cur+152:cur+156], dtype="<u4")[0]
            dlen = np.fromfile(self._fd, dtype=np.uint32, count=1)[0]
            # print("found {} bytes".format(dlen))
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

    def _write_main_header(self, md):
        ''' Write header.

            Parameters
            ----------
            md: optional
                optional md to add

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
        args = list()
        ms_keys = ['beam_center_x', 'beam_center_y', 'count_time',
                   'detector_distance', 'frame_time', 'incident_wavelength',
                   'x_pixel_size', 'y_pixel_size', 'bytes', 'nrows', 'ncols',
                   'rows_begin', 'rows_end', 'cols_begin', 'cols_end']
        '''
            16 chars : Version-COMP0001
            8 doubles : 'beam_center_x', 'beam_center_y', 'count_time',
                   'detector_distance', 'frame_time', 'incident_wavelength',
                   'x_pixel_size', 'y_pixel_size'
            7 integers : 'bytes', 'nrows', 'ncols',
                   'rows_begin', 'rows_end', 'cols_begin', 'cols_end'
        '''
        args.append("Version-COMP0001".encode())
        for key in ms_keys:
            args.append(md.get(key))
        fmt = '@16s8d7I916x'
        br = struct.pack(fmt, *args)
        # now write the md
        self._fd.write(br)

    def _read_raw(self, n):
        ''' Read from raw.
            Reads from current cursor in file.
        '''
        if n > self.Nframes:
            raise KeyError("Error, only {} frames, asked for {}"
                           .format(self.Nframes, n))
        # dlen is 4 bytes
        cur = self.frame_indexes[n]
        # dlen = np.frombuffer(self._fd[cur:cur+4], dtype="<u4")[0]
        self._fd.seek(cur, os.SEEK_SET)
        dlen = np.fromfile(self._fd, dtype=np.uint32, count=1)[0]
        cur += 4

        # pos = self._fd[cur: cur+dlen*4]
        # pos = np.frombuffer(pos, dtype='<u4')
        # self._fd.seek(cur,os.SEEK_SET)
        pos = np.fromfile(self._fd, dtype=np.uint32, count=dlen)

        cur += dlen*4
        # TODO: 2-> nbytes
        # vals = self._fd[cur: cur+dlen*self.nbytes]
        # vals = np.frombuffer(vals, dtype=self.valtype)
        # self._fd.seek(cur,os.SEEK_SET)
        vals = np.fromfile(self._fd, dtype=self.valtype, count=dlen)

        return pos, vals

    def _write_raw(self, pos, vals):
        ''' Read from raw.
            Reads from current cursor in file.
        '''
        # dlen is 4 bytes
        if len(pos) != len(vals):
            msg = "Error, len(pos) != len(vals)\n"
            msg += f"{len(pos)} != {len(vals)}"
            raise ValueError(msg)

        dlen = struct.pack("@I", len(pos))
        self._fd.write(dlen)

        pos = pos.astype(np.uint32)
        self._fd.write(pos.tobytes(order="C"))

        vals = vals.astype(self.valtype)
        self._fd.write(vals.tobytes(order="C"))

    def writeframe(self, img):
        pass

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
