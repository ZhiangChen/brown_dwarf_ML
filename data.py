#!/usr/bin/env python
"""
data.py
Zhiang Chen
Nov 22, 2019
"""

from readData import *
import copy
import numpy as np

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
        # divide the datasets
        data_nm = len(self.data)
        synthetic_data_nm = len(self.synthetic_data)
        self.train_data = copy.deepcopy(self.data[: int(data_nm*data_ratio)])
        self.test_data = copy.deepcopy(self.data[int(data_nm*data_ratio) :])
        self.train_synthetic = copy.deepcopy(self.synthetic_data[: int(synthetic_data_nm*synthetic_data_ratio)])
        self.test_synthetic = copy.deepcopy(self.synthetic_data[int(synthetic_data_nm*synthetic_data_ratio):])

    def getTrain(self, synthetic=False):
        if synthetic:
            assert len(self.train_synthetic) > 0
            return self._getTF_Data(self.train_synthetic)

        else:
            assert len(self.train_data) > 0
            return self._getTF_Data(self.train_data)

    def resampleTrain(self):
        """
        resample the real training observations from Gaussian distribution on each data point with error bars
        :return:
        """
        assert len(self.train_data) > 0
        train_data = self.resampleData(self.train_data)
        # do linear interpolation on the fly
        train_data = self.interpolateData(train_data)
        return self._getTF_Data(train_data)

    def getTest(self, synthetic=False):
        if synthetic:
            assert len(self.test_synthetic) > 0
            return self._getTF_Data(self.test_synthetic)
        else:
            assert len(self.test_data) > 0
            return self._getTF_Data(self.test_data)

    def _getTF_Data(self, data):
        flux = np.asarray([obj['flux'] for obj in data])
        teff = np.asarray([obj['teff'] for obj in data])
        logg = np.asarray([obj['logg'] for obj in data])
        label = np.swapaxes(np.vstack((teff, logg)), 0, 1)
        return flux, label

if __name__ == "__main__":
    BD = browndwarf()
    BD.loadPickles('20_BD_data.pickle', 'truncated_synthetic_data.pickle')
    # I don't want to include any real observations in the training
    BD.prepareDatasets(data_ratio=0.2, synthetic_data_ratio=0.8)
    BD.getTrain(synthetic=True)
    BD.getTest(synthetic=True)
    BD.resampleTrain()
