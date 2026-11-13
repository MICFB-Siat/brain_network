from Util import Util
import pandas as pd
import nibabel as nib
import numpy as np
from scipy.io import loadmat
import os

class HyperPara:

    def __init__(self, roi_num, c=150000, k=15, tao=1250000, kappa=12500):
        self.c = c
        self.k = k
        self.kappas = np.array([0] + [kappa for i in range(roi_num)])
        self.taos = np.array([0] + [tao for i in range(roi_num)])

class Dataloader:

    def __init__(self, csv_lh_fmri_paths, lh_gradfile_path, lh_surface_file, csv_rh_fmri_paths, rh_gradfile_path, rh_surface_file, medial_file=None):
        current_file_path = os.path.abspath(__file__)
        current_dir_path = os.path.dirname(current_file_path)
        if medial_file is None:
            medial_file = os.path.join(current_dir_path, 'tmp/fs_LR_32k_medial_mask.mat')
        self.csv_lh_fmri_paths = csv_lh_fmri_paths
        self.lh_gradfile_path = lh_gradfile_path
        self.lh_surface_file = lh_surface_file
        self.csv_rh_fmri_paths = csv_rh_fmri_paths
        self.rh_gradfile_path = rh_gradfile_path
        self.rh_surface_file = rh_surface_file
        self.medial_file = medial_file
        self.medial = np.array(loadmat(medial_file)['medial_mask'])
        self.lh_medial = self.medial[:self.medial.shape[0] // 2, 0]
        self.rh_medial = self.medial[self.medial.shape[0] // 2:, 0]

    def load_lh_fmri(self):
        df = pd.read_csv(self.csv_lh_fmri_paths, header=None)
        fmri_data_concat = np.array([])
        for (index, row) in df.iterrows():
            path = row[0]
            fmri_data = nib.load(path).get_fdata()
            fmri_data = Util.z_norm(fmri_data)
            if fmri_data_concat.size == 0:
                fmri_data_concat = fmri_data
            else:
                fmri_data_concat = np.concatenate((fmri_data_concat, fmri_data), axis=0)
        fmri_data_concat = Util.norm_norm(fmri_data_concat)
        return fmri_data_concat

    def load_lh_grad(self):
        grad_data = nib.load(self.lh_gradfile_path).get_fdata()
        grad_data = Util.norm_minmax_allmatrix(grad_data)
        return grad_data

    def load_lh_coord(self):
        lh_gii_file = nib.load(self.lh_surface_file)
        lh_coord = lh_gii_file.darrays[0].data
        lh_coord = Util.norm_norm(lh_coord.T)
        return lh_coord

    def load_lh_neighbor(self):
        lh_gii_file = nib.load(self.lh_surface_file)
        lh_neighbor = lh_gii_file.darrays[1].data
        return lh_neighbor

    def load_rh_fmri(self):
        df = pd.read_csv(self.csv_rh_fmri_paths, header=None)
        fmri_data_concat = np.array([])
        for (index, row) in df.iterrows():
            path = row[0]
            fmri_data = nib.load(path).get_fdata()
            fmri_data = Util.z_norm(fmri_data)
            if fmri_data_concat.size == 0:
                fmri_data_concat = fmri_data
            else:
                fmri_data_concat = np.concatenate((fmri_data_concat, fmri_data), axis=0)
        fmri_data_concat = Util.norm_norm(fmri_data_concat)
        return fmri_data_concat

    def load_rh_grad(self):
        grad_data = nib.load(self.rh_gradfile_path).get_fdata()
        grad_data = Util.norm_minmax_allmatrix(grad_data)
        return grad_data

    def load_rh_coord(self):
        rh_gii_file = nib.load(self.rh_surface_file)
        rh_coord = rh_gii_file.darrays[0].data
        rh_coord = Util.norm_norm(rh_coord.T)
        return rh_coord

    def load_rh_neighbor(self):
        rh_gii_file = nib.load(self.rh_surface_file)
        rh_neighbor = rh_gii_file.darrays[1].data
        return rh_neighbor

    def load(self):
        lh_fmri = self.load_lh_fmri()
        lh_grad = self.load_lh_grad()
        lh_coord = self.load_lh_coord()
        lh_neighbor = self.load_lh_neighbor()
        rh_fmri = self.load_rh_fmri()
        rh_grad = self.load_rh_grad()
        rh_coord = self.load_rh_coord()
        rh_neighbor = self.load_rh_neighbor()
        return ((lh_fmri, lh_grad, lh_coord, lh_neighbor, self.lh_medial), (rh_fmri, rh_grad, rh_coord, rh_neighbor, self.rh_medial))
if __name__ == '__main__':
    current_file_path = os.path.abspath(__file__)
    current_dir_path = os.path.dirname(current_file_path)
    csv_lh_fmri_paths = 'lh_fmri.csv'
    lh_gradfile_path = 'lh_gradient_matrix.dscalar.nii'
    lh_surface_file = os.path.join(current_dir_path, 'tmp/S1200.L.sphere.32k_fs_LR.surf.gii')
    csv_rh_fmri_paths = 'rh_fmri.csv'
    rh_gradfile_path = 'rh_gradient_matrix.dscalar.nii'
    rh_surface_file = os.path.join(current_dir_path, 'tmp/S1200.R.sphere.32k_fs_LR.surf.gii')
    dataloader = Dataloader(csv_lh_fmri_paths, lh_gradfile_path, lh_surface_file, csv_rh_fmri_paths, rh_gradfile_path, rh_surface_file)
    (lh, rh) = dataloader.load()
