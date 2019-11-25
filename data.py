#!/usr/bin/env python
"""
data.py
Zhiang Chen
Nov 22, 2019
"""

from readData import *
import copy
import numpy as np
from random import shuffle

class browndwarf(dataReader):
    def __init__(self, pickle_path=None, txt_path=None):
        super(browndwarf, self).__init__(pickle_path, txt_path)

    def loadPickles(self, data_pickle, synthetic_pickle):
        self.data= self.read_Pickle(data_pickle)
        self.synthetic_data = self.read_Pickle(synthetic_pickle)

    def prepareDatasets(self, data_ratio, synthetic_data_ratio):
        assert (data_ratio >=0) & (data_ratio <=1)
        assert (synthetic_data_ratio >=0) & (synthetic_data_ratio <=1)
        # linearly interpolate the synthetic data
        # can't linearly interpolate the real data because the resampling depends on the error bars while linear
        # interpolation will eliminate the error bars
        self.synthetic_data = self.interpolateData(self.synthetic_data)
        # shuffle data
        shuffle(self.data)
        shuffle(self.synthetic_data)
        # divide the datasets
        data_nm = len(self.data)
        synthetic_data_nm = len(self.synthetic_data)
        self.train_data = copy.deepcopy(self.data[: int(data_nm*data_ratio)])
        self.test_data = copy.deepcopy(self.data[int(data_nm*data_ratio) :])
        self.train_synthetic = copy.deepcopy(self.synthetic_data[: int(synthetic_data_nm*synthetic_data_ratio)])
        self.test_synthetic = copy.deepcopy(self.synthetic_data[int(synthetic_data_nm*synthetic_data_ratio):])

    def getTrain(self, synthetic=False):
        """
        get train data
        :param synthetic:
        :return: flux: ndarray(None, 200), label(teff, logg): ndarray(None, 2)
        """
        if synthetic:
            assert len(self.train_synthetic) > 0
            data = self._scaleSyntheticData(self.train_synthetic)
            return self._getTF_Data(data)

        else:
            assert len(self.train_data) > 0
            """!!!!!!!"""
            # scale real data
            return self._getTF_Data(self.train_data)

    def resampleTrain(self):
        """
        resample the real training observations from Gaussian distribution on each data point with error bars
        :return: flux: ndarray(None, 200), label(teff, logg): ndarray(None, 2)
        """
        assert len(self.train_data) > 0
        train_data = self.resampleData(self.train_data)
        # do linear interpolation on the fly
        train_data = self.interpolateData(train_data)
        """!!!!!!!"""
        # scale real data
        return self._getTF_Data(train_data)

    def getTest(self, synthetic=False):
        """
        get test data
        :param synthetic:
        :return: flux: ndarray(None, 200), label(teff, logg): ndarray(None, 2)
        """
        if synthetic:
            assert len(self.test_synthetic) > 0
            data = self._scaleSyntheticData(self.train_synthetic)
            return self._getTF_Data(data)
        else:
            assert len(self.test_data) > 0
            """!!!!!!!"""
            # scale real data
            return self._getTF_Data(self.test_data)

    def splitData(self, flux, labels, split=0.8):
        nm = flux.shape[0]
        p = int(nm*split)
        flux_1 = flux[: p, :]
        labels_1 = labels[: p, :]
        flux_2 = flux[p:, :]
        labels_2 = labels[p:, :]
        return flux_1, labels_1, flux_2, labels_2

    def _getTF_Data(self, data):
        flux = np.asarray([obj['flux'] for obj in data])
        teff = np.asarray([obj['teff'] for obj in data])
        logg = np.asarray([obj['logg'] for obj in data])
        label = np.swapaxes(np.vstack((teff, logg)), 0, 1)
        return flux, label

    def _scaleSyntheticData(self, data_list):
        new_data_list = copy.deepcopy(data_list)
        for data in new_data_list:
            data['flux'] = self._logTransform(data['flux']*1e-11, 200)
            data['teff'] = 1.0/650 * data['teff'] - 300.0/650
            data['logg'] = 1.0/2.5 * data['logg'] - 3.0/2.5
        return new_data_list

    def _scaleData(self, data_list):
        new_data_list = copy.deepcopy(data_list)
        for data in new_data_list:
            pass

    def _logTransform(self, data, base):
        # logarithm base change rule
        return np.log(data + 1.0/base)/np.log(base) + 1

if __name__ == "__main__":
    BD = browndwarf()
    BD.loadPickles('20_BD_data.pickle', 'truncated_synthetic_data.pickle')
    # I don't want to include any real observations in the training
    BD.prepareDatasets(data_ratio=1.0, synthetic_data_ratio=0.8)
    # Inspect input scaling
    # real observation

    # synthetic data
    """
    data_list = BD._scaleSyntheticData(BD.synthetic_data)
    loggs = []
    for data in data_list:
        loggs.append(data['logg'])
    print(max(loggs))
    print(min(loggs))
    plt.hist(np.sort(loggs), bins=10)
    plt.show()

    max_flux = []
    for data in data_list:
        max_flux.append(data['flux'].max())
    plt.hist(np.sort(max_flux), bins=10)
    plt.show()

    teffs = []
    for data in data_list:
        teffs.append(data['teff'])
    plt.hist(np.sort(teffs), bins=10)
    plt.show()
    """
    flux, labels = BD.getTrain(synthetic=True)
    print(np.max(labels, axis=0))
    print(np.min(labels, axis=0))

    for f in flux:
        plt.plot(BD.synthetic_data[0]['wavelength'], f)
    plt.show()