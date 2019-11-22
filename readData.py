#!/usr/bin/env python
"""
readData.py
Zhiang Chen
Nov 17, 2019
"""

from docx import Document
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import copy

class dataReader(object):
    def __init__(self, pickle_path=None, txt_path=None):
        self.pickle_path = pickle_path
        self.txt_path = txt_path
        self.data = []
        self.synthetic_data = []

    def read_PickleData(self):
        files = os.listdir(self.pickle_path)
        label_file = self.pickle_path + [f for f in files if f.endswith(".docx")][0]
        doc = Document(open(label_file))
        table = doc.tables[0]

        keys = None
        for i, row in enumerate(table.rows):
            text = [cell.text for cell in row.cells]

            # Establish the mapping based on the first row
            # headers; these will become the keys of our dictionary
            if i == 0:
                text[1] = "logg"
                text[2] = "teff"
                text[0] = "name"
                keys = tuple(text)
                continue

            # Construct a dictionary for this row, mapping
            # keys to values for this row
            text[0] = str(text[0])
            text[1] = float(text[1])
            text[2] = float(text[2])
            row_data = dict(zip(keys, text))
            pickle_file_name = self.pickle_path + "data_" + text[0] + ".pic"
            self._addData(pickle_file_name, row_data)
            self.data.append(row_data)

    def read_TXTData(self):
        files = [f for f in os.listdir(self.txt_path) if f.endswith('spectrum.txt')]
        for f in files:
            text_file = open(self.txt_path + f, 'rb')
            text = text_file.read()
            text_file.close()
            text_list = text.split('\n')[1:-1]
            wv = np.array([float(t.split(" ")[0]) for t in text_list])
            flux = np.array([float(t.split(" ")[1]) for t in text_list])
            if wv[0] - wv[-1] > 1:
                wv = np.flip(wv)
                flux = np.flip(flux)
            label = f.split("spectrum.txt")[0].split("_")
            teff, logg = label[1], label[3]
            data = dict()
            data['wavelength'] = wv
            data['flux'] = flux
            data['teff'] = teff
            data['logg'] = logg
            self.synthetic_data.append(data)


    def _addData(self, pickle_file_name, dic):
        with open(pickle_file_name, 'rb') as pickle_file:
            d = pickle.load(pickle_file)
            dic["wavelength"] = d[0]
            dic["flux"] = d[1]
            dic["err"] = d[2]


    def save_Pickle(self, f_name, data):
        pickle.dump(data, open(f_name, 'wb'))

    def read_Pickle(self, f_name):
        with open(f_name, 'rb') as pickle_file:
            return pickle.load(pickle_file)

    def truncateData(self, low_wv=0.95, high_wv=2.4):
        sampled_data = []
        for obj in self.data:
            wv = obj['wavelength']
            flux = obj['flux']
            err = obj['err']
            loc = np.where((wv < high_wv) & (wv > low_wv))[0]
            obj['wavelength'] = wv[loc]
            obj['flux'] = flux[loc]
            obj['err'] = err[loc]
            sampled_data.append(obj)
        self.data = sampled_data

    def truncateSyntheticData(self, low_wv=0.95, high_wv=2.4):
        sampled_data = []
        for obj in self.synthetic_data:
            wv = obj['wavelength']
            flux = obj['flux']
            loc = np.where((wv < high_wv) & (wv > low_wv))[0]
            obj['wavelength'] = wv[loc]
            obj['flux'] = flux[loc]
            sampled_data.append(obj)
        self.synthetic_data = sampled_data

    def interpolateData(self, data_list, nm=200, low_wv=0.95, high_wv=2.4):
        new_wv = np.linspace(low_wv, high_wv, nm+1)[:nm]
        new_data_list = copy.deepcopy(data_list)
        for data in new_data_list:
            wv = data['wavelength']
            flux = data['flux']
            new_flux = np.interp(new_wv, wv, flux)
            data['wavelength'] = new_wv
            data['flux'] = new_flux
        return new_data_list

    def resampleData(self, data_list):
        new_data_list = copy.deepcopy(data_list)
        for data in new_data_list:
            wv = data['wavelength']
            flux = data['flux']
            err = data['err']
            new_flux = np.random.normal(flux, err)
            del data['err']
            data['flux'] = new_flux
        return new_data_list

if __name__ == "__main__":
    pickle_path = "./20browndwarf/"
    txt_path = "./cold_BD_grid/"
    dr = dataReader(pickle_path, txt_path)
    # read from original data and generate pickle files
    """
    dr.read_PickleData()
    dr.save_Pickle("20_BD_data.pickle", dr.data)
    dr.read_TXTData()
    dr.save_Pickle("synthetic_data.pickle", dr.synthetic_data)
    """

    # save truncated data
    """
    dr.synthetic_data = dr.read_Pickle("synthetic_data.pickle")
    dr.truncateSyntheticData()
    dr.save_Pickle("truncated_synthetic_data.pickle", dr.synthetic_data)
    """

    # read from pickles
    dr.data = dr.read_Pickle("20_BD_data.pickle")
    dr.synthetic_data = dr.read_Pickle("truncated_synthetic_data.pickle")

    print(len(dr.synthetic_data[0]['wavelength']))
    print(dr.synthetic_data[0]['wavelength'])
    plt.plot(dr.synthetic_data[0]['wavelength'], dr.synthetic_data[0]['flux'])
    data = dr.interpolateData(dr.synthetic_data)
    plt.plot(data[0]['wavelength'], data[0]['flux'], label='linear interp')
    plt.show()


    # plot original, resampled, and interpolation data
    """
    dr.truncateData()
    data = dr.interpolateData(dr.data)
    plt.plot(dr.data[19]['wavelength'], dr.data[19]['flux'], label='original')
    plt.plot(data[19]['wavelength'], data[19]['flux'], label='linear interp')
    data = dr.resampleData(dr.data)
    plt.plot(data[19]['wavelength'], data[19]['flux'], label='resample')
    plt.legend()
    plt.show()
    """

    # plot all objects
    """
    plt.figure()
    for i in range(20):
        print(len(dr.data[i]['wavelength']))
        #if len(dr.data[i]['wavelength'])>150:
        plt.plot(dr.data[i]['wavelength'], dr.data[i]['flux'], label=str(dr.data[i]['teff']) + "_" + str(dr.data[i]['logg']))
    #plt.figure()
    #plt.plot(dr.data[-1]['wavelength'], dr.data[-1]['flux'])
    print(dr.data[0]['wavelength'] - dr.data[1]['wavelength'])
    #print(dr.data[19]['wavelength'])
    plt.legend()
    plt.show()
    """

