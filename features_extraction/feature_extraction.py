# -*- coding: utf-8 -*-
#
# comparison_schemes.py
#

"""
Features extraction script.
"""

__author__ = 'Ahmed Albuni'
__email__ = 'ahmed.albuni@gmail.com'


import argparse
import logging
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import SimpleITK as sitk
import six
from radiomics import (firstorder, glcm, gldm, glrlm, glszm, ngtdm, shape,
                       shape2D)
from tqdm import tqdm
from csv import DictWriter

parser = argparse.ArgumentParser(description='Features extraction')
parser.add_argument("-file", type=str, help='CSV parameters file name and '
                                            'path')
parser.add_argument("-glcm_distance", type=str, help='list of distances, '
                                                     'comma separated. '
                                                     'default: 1')
parser.add_argument("-ngtdm_distance", type=str, help='list of distances, '
                                                      'comma separated. '
                                                      'default 1')
parser.add_argument("-gldm_distance", type=str, help='list of distances, '
                                                     'comma separated. '
                                                     'default 1')
parser.add_argument("-gldm_a", type=int, help='Cutoff value for dependence, '
                                              'default: 0')

FEATURES_LIST = ('shape', 'first_order', 'glszm', 'glrlm', 'ngtdm',
                 'gldm', 'glcm')


def extract_radiomics_features(features_list, bin_width, images_path,
                               masks_path=None,
                               glcm_distance=None,
                               ngtdm_distance=None,
                               gldm_distance=None,
                               gldm_a=0,
                               output_file_name='output'):
    """
    :param features_list: list of features to be extracted
    :param bin_width:
    :param images_path: The path that contains the images
    :param masks_path: The path of the masks, masks name should match the
    images names
    :param glcm_distance: A list of distances for GLCM calculations,
    default is [1]
    :param ngtdm_distance: List of integers. This specifies the distances
     between the center voxel and the neighbor, for which angles should be
      generated.
    :param gldm_distance: List of integers. This specifies the distances
     between the center voxel and the neighbor, for which angles should be
      generated.
    :param gldm_a:  integer, α cutoff value for dependence.
    A neighbouring voxel with gray level j is considered
    dependent on center voxel with gray level i if |i−j|≤α
    :param output_file_name: Name of the output csv file
    :return:
    """
    if glcm_distance is None:
        glcm_distance = [1]
    if ngtdm_distance is None:
        ngtdm_distance = [1]
    if gldm_distance is None:
        gldm_distance = [1]

    list_of_images = [f for f in listdir(images_path) if isfile(join(
        images_path, f))]

    for i, img in tqdm(enumerate(list_of_images), total=len(list_of_images),
                       unit="files"):
        image_name = images_path+img
        image = sitk.ReadImage(image_name, sitk.sitkUInt8)

        row = dict()
        row['Name'] = img
        if i == 0:
            columns = ['Name']

        if masks_path is None:
            mask = np.zeros((sitk.GetArrayFromImage(image)).shape, int) + 1
            mask = sitk.GetImageFromArray(mask)

        else:
            mask_name = masks_path+img
            mask = sitk.ReadImage(mask_name, sitk.sitkUInt8)
            # Shape features applied only when the mask is provided
            if 'shape' in features_list:
                if len((sitk.GetArrayFromImage(image)).shape) == 2:
                    shape_2d_f = shape2D.RadiomicsShape2D(image, mask,
                                                          binWidth=bin_width)
                    row.update(get_selected_features(shape_2d_f))
                else:
                    shape_f = shape.RadiomicsShape(image, mask,
                                                   binWidth=bin_width)
                    row.update(get_selected_features(shape_f))

        if 'first_order' in features_list:
            f_o_f = firstorder.RadiomicsFirstOrder(image, mask,
                                                   binWidth=bin_width)
            row.update(get_selected_features(f_o_f))
        if 'glszm' in features_list:
            glszm_f = glszm.RadiomicsGLSZM(image, mask, binWidth=bin_width)
            row.update(get_selected_features(glszm_f))
        if 'glrlm' in features_list:
            glrlm_f = glrlm.RadiomicsGLRLM(image, mask, binWidth=bin_width)
            row.update(get_selected_features(glrlm_f))
        if 'ngtdm' in features_list:
            for d in ngtdm_distance:
                ngtdm_f = ngtdm.RadiomicsNGTDM(image, mask, distances=[d],
                                               binWidth=bin_width)
                row.update(get_selected_features(ngtdm_f,
                                                 additional_param='_d_' +
                                                                  str(d)))
        if 'gldm' in features_list:
            for d in gldm_distance:
                gldm_f = gldm.RadiomicsGLDM(image, mask, distances=[d],
                                            gldm_a=gldm_a,
                                            binWidth=bin_width)
                row.update(get_selected_features(gldm_f,
                                                 additional_param='_d_' +
                                                                  str(d)))
        if 'glcm' in features_list:
            for d in glcm_distance:
                glcm_f = glcm.RadiomicsGLCM(image, mask, distances=[d],
                                            binWidth=bin_width)
                row.update(get_selected_features(glcm_f,
                                                 additional_param='_d_' +
                                                                  str(d)))
        if i == 0:
            create_output_file(output_file_name + '.csv', row.keys())
        add_row_of_data(output_file_name+'.csv', row.keys(), row)


def create_output_file(file_name, columns):
    with open(file_name, "w", newline='') as f:
        writer = DictWriter(f, fieldnames=columns)
        writer.writeheader()


def add_row_of_data(file_name, columns, row):
    with open(file_name, "a", newline='') as f:
        writer = DictWriter(f, fieldnames=columns)
        writer.writerow(row)


def get_selected_features(selected_feature, additional_param=None):
    selected_feature.execute()
    data = {}
    for (key, val) in six.iteritems(selected_feature.featureValues):
        if additional_param is not None:
            key = key + additional_param
        data[key] = val

    return data


if __name__ == '__main__':

    logging.disable(logging.CRITICAL)

    args = parser.parse_args()
    glcm_d = args.glcm_distance
    if glcm_d is not None:
        glcm_d = glcm_d.split(',')
    ngtdm_d = args.ngtdm_distance
    if ngtdm_d is not None:
        ngtdm_d = ngtdm_d.split(',')
    gldm_d = args.gldm_distance
    if gldm_d is not None:
        gldm_d = gldm_d.split(',')

    gldm_a = args.gldm_a
    if gldm_a is None:
        gldm_a = 0

    f_list = pd.read_csv(args.file)

    for index, row in f_list.iterrows():
        print('Output file: ', row['output_file_name'])
        feature = []
        for f in FEATURES_LIST:
            if row[f] == 1:
                feature.append(f)
        if type(row['mask_dir']) is not str:
            mask_path = None
        else:
            mask_path = row['mask_dir']
        extract_radiomics_features(feature, row['bin_width'],
                                   row['image_dir'], mask_path,
                                   output_file_name=row['output_file_name'],
                                   glcm_distance=glcm_d,
                                   ngtdm_distance=ngtdm_d,
                                   gldm_distance=gldm_d,
                                   gldm_a=gldm_a)
