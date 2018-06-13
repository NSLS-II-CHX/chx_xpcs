#################
# this is for the xpcs config file, used at APS
# for the xpcs-eigen library developed by Faisal Khan at APS
#################
import os
import yaml
import h5py
import numpy as np

from ..eiger.eiger import get_header_dict, get_valid_keys

def make_config_file(filename, dqmap, sqmap,
                            out_filename="out.hd5", beg=1, end=None,
                            stride_frames=1, avg_frames=1,
                            delays_per_level=4,
                            eiger_version="v1.3.0"
                            ):
    '''
        Make an hdf5 configuration file from the filename.

        This configuration file is used for the xpcs-eigen library.

        Parameters
        ----------

        filename : the filename of the EIGER master file

        dqmap: 2d np.ndarray
            the dynamic q/phi partitions
            Each partition must contain a unique integer id from 0 to number of
            partitions.
            The mapping from these integer id's back to q/phi or other
            coordinate system is up to the user

        dqmap: 2d np.ndarray
            the static q/phi partitions
            Each partition must contain a unique integer id from 0 to number of
            partitions.
            The mapping from these integer id's back to q/phi or other
            coordinate system is up to the user

        out_filename : the output filename for the hdf5 configuration file

        beg: int, optional
            The beginning frame to analyze

        end: int or None, optional
            The end frame to analyze
            if None, then use the last frame

        stride_frames: int, optional
            skip every nth frame where "n" is the stride_frames
            stride is performed before average

        avg_frames: int, optional
            average every n frames together where "n" is avg_frames
            the striding is performed before the average (see stride_frames)

        delays_per_level: int, optional
            the number of delays per level for the multitau analysis

        eiger_version: the version of the EIGER file format used

        Returns
        -------
        xpcs_config : an XPCS config file
    '''
    with h5py.File(out_filename, "w") as fout:
        dset_keys, dims_per_key = get_valid_keys(filename, version=eiger_version)
        Nkeys = len(dset_keys)
        imgs_per_key = dims_per_key[0]
        dims = dims_per_key[1:]

        mask = (dqmap > 0).astype(int)

        if end is None:
            end = Nkeys*imgs_per_key

        # the output path into the out hdf5 file
        fout['/xpcs/output_data'] = b'/exchange'

        fout['/xpcs/compression'] = 'ENABLED'

        # we don't need a flatfield
        fout['/xpcs/flatfield_enabled'] = b'DISABLED'

        # is this version right?
        fout['/xpcs/Version'] = b'0.5'

        # should this be here?
        fout['/xpcs/analysis_type'] = b'Multitau'

        # average frames by this number. we want to keep all frames
        # stride performed before average
        fout['/xpcs/stride_frames'] = np.array([[stride_frames]])
        fout['/xpcs/avg_frames'] = np.array([[avg_frames]])

        # disable the blemish file
        fout['/xpcs/blemish_enabled'] = b'DISABLED' # ENABLED?

        # no darks enabled right now
        fout['/xpcs/dark_begin'] = np.array([[ 0.]])
        fout['/xpcs/dark_begin_todo'] = np.array([[ 0.]])
        fout['/xpcs/dark_end'] = np.array([[ 0.]])
        fout['/xpcs/dark_end_todo'] = np.array([[ 0.]])

        # data begin and end and todo (to run)
        fout['/xpcs/data_begin'] = np.array([[1]])
        fout['/xpcs/data_begin_todo'] = np.array([[beg]])
        fout['/xpcs/data_end'] = np.array([[end]])
        fout['/xpcs/data_end_todo'] = np.array([[end]])

        # mask
        fout['/xpcs/mask'] = mask

        # multitau stuff
        fout['/xpcs/delays_per_level'] = np.array([[delays_per_level]])


        # the partition maps
        fout['/xpcs/dqmap'] =  dqmap
        fout['/xpcs/sqmap'] =  sqmap


        header = get_header_dict(filename, dims, version=eiger_version)
        # get x y dimension from EIGER file
        fout['/measurement/instrument/detector/x_dimension'] = dims[1]
        fout['/measurement/instrument/detector/y_dimension'] = dims[0]
        fout['/measurement/instrument/detector/x_pixel_size'] = \
                header['x_pixel_size']
        fout['/measurement/instrument/detector/y_pixel_size'] = \
                header['y_pixel_size']
        # not used for eiger
        fout['/measurement/instrument/detector/adu_per_photon'] = 1
        fout['/measurement/instrument/detector/exposure_time'] = 1
        fout['/measurement/instrument/detector/efficiency'] = 1
        fout['/measurement/instrument/detector/distance'] = 1
        fout['/measurement/instrument/source_begin/beam_intensity_transmitted'] = 1
        fout['/measurement/sample/thickness'] = 1
        fout['/measurement/instrument/detector/flatfield'] = 1

        # how many frames to group for partitioning
        # this will compute an extra statistic per this number of frames
        # average versus time etc
        fout['/xpcs/static_mean_window_size'] = [[51]]
        fout['/xpcs/input_file_local'] = ""

        #####################################################################
        # variables I think we don't need or I'm not sure about below here: #
        #####################################################################

        if False:
            # big array of philist and qlist for dynamic ("d") partitions
            # these are NOT needed correct? (since we have the qmaps and phimaps
            # already)
            fout['/xpcs/dphilist'] = np.array([[]])
            fout['/xpcs/dqlist'] = np.array([[]])
            # what is this?
            fout['/xpcs/Normalize_by_FrameSum'] = np.array([[0]])
            # what is this? do we need this?
            fout['/xpcs/dphispan'] = np.array([[-268.8913269, 89.53925323]])
            fout['/xpcs/dqspan'] = np.array([[ ]])

            # I assume thes can be extracted from the dqmap, sqmap
            # dynamic partition number
            fout['/xpcs/dnophi'] = np.array([[dnophis]])
            fout['/xpcs/dnoq'] = np.array([[dnoqs]])
            # static partition number
            fout['/xpcs/snophi'] = np.array([[snophis]])
            fout['/xpcs/snoq'] = np.array([[ snoqs]])


            # what is this exactly?
            fout['/xpcs/dynamic_mean_window_size'] = np.array([[51]])
            # not needed since we'll specify this later
            fout['/xpcs/input_file_local'] = ""
            fout['/xpcs/input_file_remote'] = ""

            # not needed for this method I assume
            fout['/xpcs/kinetics'] = b'DISABLED'
            # not needed since we have a counting detector
            # ( no need for low level discriminator)
            fout['/xpcs/lld'] = np.array([[ 0.]])

            # is this needed?
            fout['/xpcs/normalization_method'] = b'TRANSMITTED'

            # what is this exactly? I leave as 1 here
            fout['/xpcs/batches'] = [[1]]

            # I assume this will be done as a parameter into the analysis
            # these not needed correct?
            fout['/xpcs/output_file_local'] = ""
            fout['/xpcs/output_file_remote'] = ""
            fout['/xpcs/qmap_hdf5_filename'] = ""
            fout['/xpcs/sigma'] = [[ 0.]]
            fout['/xpcs/specfile'] = ""
            fout['/xpcs/specscan_dark_number'] = [[ 0.]]
            fout['/xpcs/specscan_data_number'] = [[775]]
            fout['/xpcs/sphilist'] = [[]]
            fout['/xpcs/sphispan'] = [[-268.8913269, 89.53925323]]
            fout['/xpcs/sqlist'] = [[]]
            fout['/xpcs/sqspan'] = [[ ]]
            fout['/xpcs/swbinX'] = [[1]]
            fout['/xpcs/swbinY'] = [[1]]
