import h5py
import numpy as np
from tqdm import tqdm
from .eiger import get_header_binary, get_valid_keys


def compress_file(filename, outfile="out.bin", version="v1.3.0", mask=None,
                  verbose=False):
    '''
        Compress an EIGER hdf5 file into a BNL Multifile compressed format.

        Parameters
        ----------
        filename : str
            The filename of the EIGER file
        outfile: str, optional
        version : str, optional
            The version string of the hdf5 file
            Currently, only major changes occurred from the change to 1.3.0
            So just make sure "new" files have a version number string
            according to this format greater than this number and vice versa
            for older files
    '''

    # open and close file, figure out what the valid keys are
    dset_keys, dims_per_key = get_valid_keys(filename, version=version)

    # Nkeys = len(dset_keys)
    nimgs = dims_per_key[0]
    dims = dims_per_key[1:]

    # dset_min = 1
    # dset_max = Nkeys+1

    arr = np.zeros(dims, dtype=np.uint16)

    # re-open file and close again, get header
    header = get_header_binary(filename, dims, version=version)

    # open the output file, start writing
    fout = open(outfile, "wb")
    fout.write(header)

    f = h5py.File(filename)

    for dset_key in tqdm(dset_keys):
        if verbose:
            print("reading dataset {}".format(dset_key))
        for j in range(nimgs):
            dset = f[dset_key]
            # this is an important trick to ensure the reading is blazingly
            # fast. Doing this incorrectly can result in a significant
            # reduction in performance! At least a factor of 10!
            dset.read_direct(arr, np.s_[j, :, :])

            # test = np.array(f[key][j])
            # TODO : Here we should use our own conditions to test
            if mask is not None:
                w, = np.where(((arr*mask).ravel() > 0))
            else:
                w, = np.where((arr.ravel() > 0))

            # print("found {} items".format(len(w[0])))
            fout.write(np.uint32(len(w)))
            fout.write(w.astype(np.uint32))
            fout.write(arr.ravel()[w])

    fout.close()
    f.close()
