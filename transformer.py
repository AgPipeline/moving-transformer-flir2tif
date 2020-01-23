"""Transforms FLIR camera output to TIF file format"""

import logging
import json
import os
import math
import numpy as np
from numpy.matlib import repmat

from terrautils.spatial import geojson_to_tuples
from terrautils.formats import create_geotiff
import configuration
try:
    import transformer_class
except ImportError:
    pass


class CalibParam:
    """Class for holding calibration information
    """
    # Disabling these checks to maintain readable code
    # pylint: disable=invalid-name, too-many-instance-attributes, too-few-public-methods
    def __init__(self):
        """Initializes class instance
        """
        # pylint: disable=invalid-name
        self.calibrated = True
        self.calibrationR = 0.0
        self.calibrationB = 0.0
        self.calibrationF = 0.0
        self.calibrationJ1 = 0.0
        self.calibrationJ0 = 0.0
        self.calibrationa1 = 0.0
        self.calibrationa2 = 0.0
        self.calibrationX = 0.0
        self.calibrationb1 = 0.0
        self.calibrationb2 = 0.0


def get_calibrate_param(metadata: dict) -> CalibParam:
    """Returns an instance of the calibration class populated with data from the metadata
    Arguments:
        metadata: the metadata to find calibration information in
    """
    calib_param = CalibParam()

    try:
        if 'terraref_cleaned_metadata' in metadata:
            fixedmd = metadata['sensor_fixed_metadata']
            if fixedmd['is_calibrated'] == 'True':
                return calib_param

            calib_param.calibrated = False
            calib_param.calibrationR = float(fixedmd['calibration_R'])
            calib_param.calibrationB = float(fixedmd['calibration_B'])
            calib_param.calibrationF = float(fixedmd['calibration_F'])
            calib_param.calibrationJ1 = float(fixedmd['calibration_J1'])
            calib_param.calibrationJ0 = float(fixedmd['calibration_J0'])
            calib_param.calibrationa1 = float(fixedmd['calibration_alpha1'])
            calib_param.calibrationa2 = float(fixedmd['calibration_alpha2'])
            calib_param.calibrationX = float(fixedmd['calibration_X'])
            calib_param.calibrationb1 = float(fixedmd['calibration_beta1'])
            calib_param.calibrationb2 = float(fixedmd['calibration_beta2'])

    except KeyError:
        pass

    return calib_param


def flir_raw_to_temperature(raw_data: np.ndarray, calib_params: CalibParam) -> np.ndarray:
    """Convert flir raw data into temperature C degree, for date after September 15th
    Arguments:
        raw_data: the raw data to convert using the calibration parameters
        calib_params: the calibration values to use when converting the data
    Return:
        Returns the calibrated data
    """
    # Disabling to maintain readability
    # pylint: disable=too-many-locals, invalid-name
    R = calib_params.calibrationR
    B = calib_params.calibrationB
    F = calib_params.calibrationF
    J0 = calib_params.calibrationJ0
    J1 = calib_params.calibrationJ1

    X = calib_params.calibrationX
    a1 = calib_params.calibrationa1
    b1 = calib_params.calibrationb1
    a2 = calib_params.calibrationa2
    b2 = calib_params.calibrationb2

    H2O_K1 = 1.56
    H2O_K2 = 0.0694
    H2O_K3 = -0.000278
    H2O_K4 = 0.000000685

    H = 0.1
    T = 22.0
    D = 2.5
    E = 0.98

    K0 = 273.15

    im = raw_data

    AmbTemp = T + K0
    AtmTemp = T + K0

    H2OInGperM2 = H*math.exp(H2O_K1 + H2O_K2*T + H2O_K3*math.pow(T, 2) + H2O_K4*math.pow(T, 3))
    a1b1sqH2O = (a1+b1*math.sqrt(H2OInGperM2))
    a2b2sqH2O = (a2+b2*math.sqrt(H2OInGperM2))
    exp1 = math.exp(-math.sqrt(D/2)*a1b1sqH2O)
    exp2 = math.exp(-math.sqrt(D/2)*a2b2sqH2O)

    tao = X*exp1 + (1-X)*exp2

    obj_rad = im*E*tao

    theo_atm_rad = (R*J1/(math.exp(B/AtmTemp)-F)) + J0
    atm_rad = repmat((1-tao)*theo_atm_rad, 640, 480)

    theo_amb_refl_rad = (R*J1/(math.exp(B/AmbTemp)-F)) + J0
    amb_refl_rad = repmat((1-E)*tao*theo_amb_refl_rad, 640, 480)

    corr_pxl_val = obj_rad + atm_rad + amb_refl_rad

    pxl_temp = B/np.log(R/(corr_pxl_val-J0)*J1+F)

    return pxl_temp


def raw_data_to_temperature(raw_data: np.ndarray, metadata: dict) -> np.ndarray:
    """Converts raw data to temperature data by applying calibration
    Arguments:
        raw_data: the raw data to calibrate
        metadata: the metadata containing the calibration values
    Return:
        Returns the calibrated data
    """
    try:
        calib_params = get_calibrate_param(metadata)

        if calib_params.calibrated:
            temp_calib = raw_data/10
        else:
            temp_calib = flir_raw_to_temperature(raw_data, calib_params)

        return temp_calib
    except Exception as ex:
        logging.exception('Raw to temperature exception')
        raise ex


def flir2tif(input_paths: list, full_md: dict = None) -> dict:
    # Determine metadata and BIN file
    bin_file = None
    for f in input_paths:
        if f.endswith(".bin"):
            bin_file = f
        if f.endswith("_cleaned.json") and full_md is None:
            with open(f, 'r') as mdf:
                full_md = json.load(mdf)['content']

    # TODO: Figure out how to pass extractor details to create_geotiff in both types of pipelines
    extractor_info = None

    if full_md:
        if bin_file is not None:
            out_file = bin_file.replace(".bin", ".tif")
            gps_bounds_bin = geojson_to_tuples(full_md['spatial_metadata']['flirIrCamera']['bounding_box'])
            raw_data = np.fromfile(bin_file, np.dtype('<u2')).reshape([480, 640]).astype('float')
            raw_data = np.rot90(raw_data, 3)
            tc = raw_data_to_temperature(raw_data, full_md)
            create_geotiff(tc, gps_bounds_bin, out_file, None, False, extractor_info, full_md, compress=True)

    # Return formatted dict for simple extractor
    return {
        "metadata": {
            "files_created": [out_file]
        },
        "outputs": [out_file]
    }


def perform_process(transformer: transformer_class.Transformer, check_md: dict, transformer_md: list, full_md: list) -> dict:
    """Performs the processing of the data
    Arguments:
        transformer: instance of transformer class
    Return:
        Returns a dictionary with the results of processing
    """
    result = {}
    file_md = []

    file_list = check_md['list_files']()

    # Find the metadata we're interested in for calibration parameters
    terra_md = None
    for one_md in full_md:
        if 'terraref_cleaned_metadata' in one_md:
            terra_md = one_md
            break
    if not terra_md:
        raise RuntimeError("Unable to find TERRA REF specific metadata")

    transformer_md = transformer.generate_transformer_md()

    try:
        for one_file in file_list:
            if one_file.endswith(".bin"):
                output_filename = os.path.join(check_md['working_folder'], os.path.basename(one_file.replace('.bin', '.tif')))
                gps_bounds_bin = geojson_to_tuples(terra_md['spatial_metadata']['flirIrCamera']['bounding_box'])
                raw_data = np.fromfile(one_file, np.dtype('<u2')).reshape([480, 640]).astype('float')
                raw_data = np.rot90(raw_data, 3)
                temp_calib = raw_data_to_temperature(raw_data, terra_md)
                create_geotiff(temp_calib, gps_bounds_bin, output_filename, None, False, transformer_md, terra_md, compress=True)

                cur_md = {'path': output_filename,
                          'key': configuration.TRANSFORMER_SENSOR,
                          'metadata': {
                              'data': transformer_md
                          }}
                file_md.append(cur_md)

        result['code'] = 0
        result['file'] = file_md

    except Exception as ex:
        msg = 'Exception caught converting FLIR files'
        logging.exception(msg)
        result['code'] = -1
        result['error'] = msg + ': ' + str(ex)

    return result
