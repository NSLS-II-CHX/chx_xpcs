from .multifile import MultifileAPS, MultifileBNL

def multifileBNL2multifileAPS(fileame_in, filename_out):
    '''
       Convert an APS multifile to a BNL multifile

       Parameters
       ----------
       filename_in : str
            the name of the input filename

       filename_out : str
            the name of the output filename
    '''

    fin = MultifileAPS(filename_in)
    # get the md
    header = fin._read_header(0)

    #fout = MultifileBNL(filenam_out, md=md)

    '''
    for i in range(fin.Nframes):
        pos, vals = fin.rdrawframe(i)
        fout._write_raw(pos, vals)
    '''

def APS2BNLheader(header):
    # convert from APS to BNL header dict
    pass
