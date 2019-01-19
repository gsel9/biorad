import os
import h5py

import scipy.io as sio

from multiprocessing import cpu_count


N_JOBS = cpu_count() - 1 if cpu_count() > 1 else cpu_count()


def matlab_to_nrrd(
    source_path, target_path, modality, transform=None, masks=None, **kwargs
):
    """Convert image data from MATLAB to NRRD format.

    Args:
        source_path (str): Reference to original data.
        target_path (str): Reference to re-formatted data.
        modality (str): Image modality.
        masks (str):
        modality (str, {`mask`, `PET`, `CT`}):

    """

    global N_JOBS

    if os.path.isdir(source_path) and os.path.isdir(target_path):

        mat_rel_paths = relative_paths(source_path, target_format='.mat')
        for num, path_mat in enumerate(mat_rel_paths):

            image_data = sio.loadmat(path_mat)
            # Apply image transformation function.
            if transform is not None:
                image = transform(image_data[kwargs['modality']], **kwargs)
            else:
                image = image_data[kwargs['modality']]

            nrrd_path = _swap_format_in_path(
                path_mat, old_format='.mat', new_format='.nrrd',
                new_path=target_path
            )
            nrrd.write(nrrd_path, image)
    else:
        raise RuntimeError('Unable to locate:\n{}\nor\n{}'
                           ''.format(source_path, target_path))

    return None


def matlab_to_hdf5(
    source_path, target_path, modality, transform=None, masks=None, **kwargs
):

    global N_JOBS

    if os.path.isdir(source_path) and os.path.isdir(target_path):

        mat_rel_paths = relative_paths(source_path, target_format='.mat')
        for num, path_mat in enumerate(mat_rel_paths):
            # MATLAB .mat files are actually HDF5 files.
            image_data = sio.loadmat(path_mat)

            # Apply image transformation function.
            if transform is not None:
                image = transform(image_data[modality], **kwargs)
            else:
                image = image_data[modality]

            # Read all cases into a single dataset.
            hdf_path = _swap_format_in_path(
                path_mat, old_format='.mat', new_format='.hdf5',
                new_path=target_path
            )
    else:
        raise RuntimeError('Unable to locate:\n{}\nor\n{}'
                           ''.format(source_path, target_path))

    return None


def _swap_format_in_path(old_path, old_format, new_format, new_path=None):
    # Auxiallary function to exchange the file format of a path.

    new_fname = os.path.basename(old_path).replace(old_format, new_format)

    if new_path is None:
        return os.path.join(os.path.dirname(old_path), new_fname)
    else:
        return os.path.join(new_path, new_fname)


if __name__ == '__main__':
    from ioutil import relative_paths

    matlab_to_hdf5(
        './../../../data_source/images/ct_mat',
        './../../../data_source/images/ct_hdf',
        modality='CT'
    )
    data = h5py.File('./../../../data_source/images/ct_hdf/P070CT.hdf5', 'r')
    print(data.keys())
