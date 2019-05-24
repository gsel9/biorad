
import matplotlib.pyplot as plt
from ioutil import relative_paths
import seaborn as sns
import nrrd
import numpy as np

import fig_config as CONFIG

CONFIG.plot_setup()


DPI = 1000


def get_ct_cropped():

    paths_to_ct = relative_paths('./../../data_source/images/ct_nrrd', target_format='nrrd')
    paths_to_mask = relative_paths('./../../data_source/images/masks_nrrd', target_format='nrrd')

    ct_cropped = {}
    for num, path_to_ct in enumerate(paths_to_ct):
        ct_image, _ = nrrd.read(path_to_ct)
        mask_image, _ = nrrd.read(paths_to_mask[num])
        ct_cropped[path_to_ct] = ct_image * mask_image

    return ct_cropped


# See: http://medvis.org/2012/08/21/rainbow-colormaps-what-are-they-good-for-absolutely-nothing/
def artifact_illustrations():

    show = False
    path_to_images = '../illustrations/artifacts'

    ct_cropped = get_ct_cropped()

    # Some of the artifacts found in the CT scans of the patients exceeding 3027 GL intensity.
    samples = [
        ct_cropped['./../../data_source/images/ct_nrrd/P038CT.nrrd'][:, :, 50],
        #ct_cropped['./../../data_source/images/ct_nrrd/P109CT.nrrd'][:, :, 47],
        ct_cropped['./../../data_source/images/ct_nrrd/P164CT.nrrd'][:, :, 59],
        #ct_cropped['./../../data_source/images/ct_nrrd/P038CT.nrrd'][:, :, 57],
        ct_cropped['./../../data_source/images/ct_nrrd/P109CT.nrrd'][:, :, 48],
        #ct_cropped['./../../data_source/images/ct_nrrd/P164CT.nrrd'][:, :, 61]
    ]
    # Create common frame for images to have equal sizes.
    sizes = [np.size(image) for image in samples]
    img_shape = np.shape(samples[np.argmax(sizes)])
    for num, image in enumerate(samples):

        nrows, ncols = np.shape(image)
        frame = np.zeros(img_shape, dtype=float)
        frame[:nrows, :ncols] = image

        plt.figure()
        plt.axis('off')
        plt.imshow(
            frame, vmin=np.min(frame), vmax=np.max(frame), cmap='viridis'
        )
        _path_to_image = path_to_images + f'_{num}.pdf'
        plt.savefig(
            _path_to_image, bbox_inches='tight', transparent=True, dpi=DPI
        )
        if show:
            plt.show()



def discretization_illustrations():

    show = False
    num_bins = [8, 16, 32, 64, 128]

    modality = 'ct'
    if modality == 'pet':
        path_to_images = '../illustrations/pet_discr'
        path_to_pet = './../../data_source/images/pet_nrrd/P222PET.nrrd'
    else:
        path_to_images = '../illustrations/ct_discr'
        path_to_pet = './../../data_source/images/ct_nrrd/P222CT.nrrd'

    path_to_mask = './../../data_source/images/masks_nrrd/P222mask.nrrd'

    image, _ = nrrd.read(path_to_pet)
    mask, _ = nrrd.read(path_to_mask)

    n = 30
    image_slice = image[n, :, :]
    mask = mask[n, :, :]

    for discr in num_bins:
        bins = np.linspace(np.min(image_slice), np.max(image_slice), discr)
        discr_img = np.digitize(image_slice, bins)
        discr_img = discr_img * mask

        plt.figure()
        plt.axis('off')
        plt.imshow(
            discr_img, vmin=np.min(discr_img), vmax=np.max(discr_img),
            cmap='viridis'
        )
        _path_to_image = path_to_images + f'_{discr}.pdf'
        plt.savefig(
            _path_to_image, bbox_inches='tight', transparent=True, dpi=DPI
        )
        if show:
            plt.show()


def cropping_illustration_imgs():

    paths_to_ct = relative_paths('./../../data_source/images/ct_nrrd', target_format='nrrd')
    paths_to_pet = relative_paths('./../../data_source/images/pet_nrrd', target_format='nrrd')
    paths_to_mask = relative_paths('./../../data_source/images/masks_nrrd', target_format='nrrd')

    path_to_ct = './../../data_source/images/ct_nrrd/P222CT.nrrd'
    path_to_pet = './../../data_source/images/pet_nrrd/P222PET.nrrd'
    path_to_mask = './../../data_source/images/masks_nrrd/P222mask.nrrd'

    path_to_ct_img = '../illustrations/ct_img.pdf'
    path_to_pet_img = '../illustrations/pet_img.pdf'
    path_to_ct_cropped_img = '../illustrations/ct_cropped_img.pdf'
    path_to_pet_cropped_img = '../illustrations/pet_cropped_img.pdf'

    show = False

    ct_image, _ = nrrd.read(path_to_ct)
    pet_image, _ = nrrd.read(path_to_pet)
    mask, _ = nrrd.read(path_to_mask)

    n = 30
    ct_image = ct_image[n, :, :]
    pet_image = pet_image[n, :, :]
    mask = mask[n, :, :]

    cropped_ct = ct_image * mask
    cropped_pet = pet_image * mask

    plt.figure()
    plt.axis('off')
    plt.imshow(ct_image, vmin=np.min(ct_image), vmax=np.max(ct_image))
    plt.savefig(
        path_to_ct_img, bbox_inches='tight', transparent=True, dpi=DPI
    )
    if show:
        plt.show()

    plt.figure()
    plt.axis('off')
    plt.imshow(cropped_ct, vmin=np.min(ct_image), vmax=np.max(ct_image))
    plt.savefig(
        path_to_ct_cropped_img, bbox_inches='tight', transparent=True, dpi=DPI
    )
    if show:
        plt.show()

    plt.figure()
    plt.axis('off')
    plt.imshow(pet_image, vmin=np.min(pet_image), vmax=np.max(pet_image))
    plt.savefig(
        path_to_pet_img, bbox_inches='tight', transparent=True, dpi=DPI
    )
    if show:
        plt.show()

    plt.figure()
    plt.axis('off')
    plt.imshow(cropped_pet, vmin=np.min(pet_image), vmax=np.max(pet_image))
    plt.savefig(
        path_to_pet_cropped_img, bbox_inches='tight', transparent=True, dpi=DPI
    )
    if show:
        plt.show()


if __name__ == '__main__':
    #cropping_illustration_imgs()
    #discretization_illustrations()
    artifact_illustrations()
