import h5py
import numpy as np

import struct

EIGER_KEYS_DEFAULT = {
    # old version (less than v1.3.0)
    'wavelength_pre1.3': "entry/instrument/monochromator/wavelength",
    # new version (greater than v1.3.0)
    'wavelength_post1.3': "entry/instrument/beam/incident_wavelength",
    'beam_center_x': "entry/instrument/detector/beam_center_x",
    'beam_center_y': "entry/instrument/detector/beam_center_x",
    'count_time': "entry/instrument/detector/count_time",
    'x_pixel_size': "entry/instrument/detector/x_pixel_size",
    'y_pixel_size': "entry/instrument/detector/y_pixel_size",
    'frame_time': "entry/instrument/detector/frame_time",
}


def get_header_binary(filename, dims, version="v1.3.0"):
    '''
        Make the BNL compressed version 1.0 format header.

        This returns a serialized form of the header.
    '''
    # this is copied from default in order to allow to mutate the dict for
    # different version EIGER's
    EIGER_KEYS = EIGER_KEYS_DEFAULT.copy()

    # resolve the version number
    if version >= "v1.3.0":
        EIGER_KEYS['wavelength'] = EIGER_KEYS['wavelength_post1.3']
    else:
        # this will be very unlikely
        EIGER_KEYS['wavelength'] = EIGER_KEYS['wavelength_pre1.3']

    # print("dims is  {}".format(dims))
    # dims : image dims
    # make the header from the EIGER file
    f = h5py.File(filename)
    # read in bytes
    # header is always from zero
    # this is version 2
    header = b"Version-COMP0002"
    header += struct.pack("@d", f[EIGER_KEYS['beam_center_x']].value)
    header += struct.pack("@d", f[EIGER_KEYS['beam_center_y']].value)
    header += struct.pack("@d", f[EIGER_KEYS['count_time']].value)
    # detector_distance
    header += struct.pack("@d", 0)
    # frame time
    header += struct.pack("@d", f[EIGER_KEYS['frame_time']].value)
    # incident wavelength
    header += struct.pack("@d", f[EIGER_KEYS['wavelength']].value)
    header += struct.pack("@d", f[EIGER_KEYS['x_pixel_size']].value)
    header += struct.pack("@d", f[EIGER_KEYS['y_pixel_size']].value)
    # bytes, we choose 2 here
    header += struct.pack("@I", 2)
    # nrows
    header += struct.pack("@I", dims[0])
    # ncols
    header += struct.pack("@I", dims[1])
    # rows begin
    header += struct.pack("@I", 0)
    # rows end
    header += struct.pack("@I", dims[0])
    # cols begin
    header += struct.pack("@I", 0)
    # cols end
    header += struct.pack("@I", dims[1])

    f.close()
    header += struct.pack("@916x")
    return header


def get_header_dict(filename, dims, version="v1.3.0"):
    '''
        Make the BNL compressed version 1.0 format header.

        This returns a dict version of the header
    '''
    # this is copied from default in order to allow to mutate the dict for
    # different version EIGER's
    EIGER_KEYS = EIGER_KEYS_DEFAULT.copy()

    # resolve the version number
    if version >= "v1.3.0":
        EIGER_KEYS['wavelength'] = EIGER_KEYS['wavelength_post1.3']
    else:
        # this will be very unlikely
        EIGER_KEYS['wavelength'] = EIGER_KEYS['wavelength_pre1.3']

    print("dims is  {}".format(dims))
    # dims : image dims
    # make the header from the EIGER file
    f = h5py.File(filename)
    # read in bytes
    # header is always from zero
    # this is version 2
    header = dict()
    header['version'] = "Version-COMP0002"
    header['beam_center_x'] = f[EIGER_KEYS['beam_center_x']].value
    header['beam_center_y'] = f[EIGER_KEYS['beam_center_y']].value
    header['count_time'] = f[EIGER_KEYS['count_time']].value
    # detector_distance
    header['detector_distance'] = 0
    # frame time
    header['frame_time'] = f[EIGER_KEYS['frame_time']].value
    # incident wavelength
    header['wavelength'] = f[EIGER_KEYS['wavelength']].value
    header['x_pixel_size'] = f[EIGER_KEYS['x_pixel_size']].value
    header['y_pixel_size'] = f[EIGER_KEYS['y_pixel_size']].value
    # nrows
    header['nrows'] = dims[0]
    # ncols
    header['ncols'] = dims[1]
    # rows begin
    header['rows_begin'] = 0
    # rows end
    header['rows_end'] = dims[0]
    # cols begin
    header['cols_begin'] = 0
    # cols end
    header['cols_end'] = dims[1]
    return header


def get_valid_keys(filename, version="v1.3.0"):
    # the prefix of the data keys. Currently the same for all versions
    dset_pref = "data_"

    f = h5py.File(filename)

    # resolve the version number
    if version >= "v1.3.0":
        dset_root = "/entry/data"
    else:
        # this will be very unlikely
        dset_root = "/entry"

    # get the range of data sets byt just inspecting the keys
    dset_keys = list(f[dset_root].keys())

    dset_keys = [dset_key for dset_key in dset_keys if
                 dset_key.startswith(dset_pref)]
    dset_keys.sort()

    # dset key validation (not necessarily needed)
    # verify numbers are all in ascending order. If one is out of sequence,
    # then there likely is an issue
    Nkeys = len(dset_keys)
    nums = np.array([int(dset_key[5:]) for dset_key in dset_keys])

    if version >= "v1.3.0":
        key_range = np.arange(1, Nkeys+1)
    else:
        key_range = np.arange(0, Nkeys)

    if not np.allclose(nums, key_range):
        msg = "Error, keys don't increase in ascending"
        msg += " order from 1 to {}".format(Nkeys)
        raise ValueError(msg)
    dset_keys = [dset_root + "/" + dset_key for dset_key in dset_keys]

    # get data set shape and number images from first entry
    dset = f[dset_keys[0]]
    dims_per_key = dset.shape
    # we should check all other keys have the same shape
    # but I decide not to (we might want to analyze incomplete data sets.
    # It would be a shame to err out on those here.)

    f.close()

    return dset_keys, dims_per_key
