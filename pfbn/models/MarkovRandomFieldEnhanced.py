from Util import Util
from scipy.special import iv
import numpy as np
import gco
from Util import Util
from dataLoader import Dataloader, HyperPara
import random
import os
from scipy.special import iv as besseli
import copy
from scipy.io import loadmat
import nibabel as nib
import time

class MarkovRandomFieldEnhanced:

    def __init__(self, dataset):
        t1 = time.time()
        self.dataset = dataset
        self.roi_num = 100
        self.hyper_para = HyperPara(roi_num=self.roi_num)
        self.ref_template = np.load(os.path.join(dataset, 'MRF_Prepare', 'ref_template_enhanced.npy'), mmap_mode='r')
        self.ref_index = np.load(os.path.join(dataset, 'MRF_Prepare', 'index.npy'), mmap_mode='r')
        self.fmri_list = [fmri for fmri in os.listdir(self.dataset) if fmri.endswith('.dtseries.nii')]
        if self.fmri_list:
            self.time_len = nib.load(os.path.join(self.dataset, self.fmri_list[0])).get_fdata().shape[0]
        else:
            fmri_avg_path = os.path.join(self.dataset, 'fmri_avg.npy')
            if os.path.isfile(fmri_avg_path):
                self.time_len = np.load(fmri_avg_path, mmap_mode='r').shape[0]
            else:
                raise FileNotFoundError(f'无 dtseries 且无 fmri_avg.npy，无法获取 time_len: {dataset}')
        if dataset[-2:] == 'lh':
            self.pre_dot_acti_mat = np.load(os.path.join(dataset, 'MRF_Prepare', 'lh_pre_dot_acti_mat.npy'), mmap_mode='r').astype(np.float32)
            self.edges = np.load(os.path.join(dataset, 'MRF_Prepare', 'lh_edges.npy'), mmap_mode='r').astype(np.int32)
            self.grad = nib.load(os.path.join(dataset, 'MRF_Prepare', 'lh_gradient_matrix.dscalar.nii')).get_fdata()
            self.pre_dot_coord_mat = np.load(os.path.join(dataset, 'MRF_Prepare', 'lh_pre_dot_coord_mat.npy'), mmap_mode='r').astype(np.float32)
            self.label = Util.rand_intarr_generate(29696, 1, self.roi_num)
            self.medial = np.array(loadmat(os.path.join(dataset, 'MRF_Prepare', 'fs_LR_32k_medial_mask.mat'))['medial_mask'])
            self.medial = self.medial[:self.medial.shape[0] // 2, 0]
            self.vox_num = 29696
        else:
            self.pre_dot_acti_mat = np.load(os.path.join(dataset, 'MRF_Prepare', 'rh_pre_dot_acti_mat.npy'), mmap_mode='r').astype(np.float32)
            self.edges = np.load(os.path.join(dataset, 'MRF_Prepare', 'rh_edges.npy'), mmap_mode='r').astype(np.int32)
            self.grad = nib.load(os.path.join(dataset, 'MRF_Prepare', 'rh_gradient_matrix.dscalar.nii')).get_fdata()
            self.pre_dot_coord_mat = np.load(os.path.join(dataset, 'MRF_Prepare', 'rh_pre_dot_coord_mat.npy'), mmap_mode='r').astype(np.float32)
            self.label = Util.rand_intarr_generate(29716, 1, self.roi_num)
            self.medial = np.array(loadmat(os.path.join(dataset, 'MRF_Prepare', 'fs_LR_32k_medial_mask.mat'))['medial_mask'])
            self.medial = self.medial[self.medial.shape[0] // 2:, 0]
            self.vox_num = 29716
        self.label[self.ref_index] = self.ref_template
        t2 = time.time()
        print('数据加载耗时为: {0}'.format(t2 - t1))

        def index_medial2non():
            nonmedial_index = 0
            mapper = {}
            for medial_index in range(len(self.medial)):
                if self.medial[medial_index] == 1:
                    mapper[medial_index] = nonmedial_index
                    nonmedial_index += 1
            return mapper
        self.index_medial2non = index_medial2non()
        self.index_non2medial = {value: key for (key, value) in self.index_medial2non.items()}

        def map_value(x):
            return self.index_non2medial.get(x, x)
        vectorized_map = np.vectorize(map_value)
        self.pre_edgeweight = np.zeros((self.edges.shape[0],))
        grad_values_1 = self.grad[vectorized_map(self.edges[:, 0]), vectorized_map(self.edges[:, 1])]
        grad_values_2 = self.grad[vectorized_map(self.edges[:, 1]), vectorized_map(self.edges[:, 0])]
        grad_avg = (grad_values_1 + grad_values_2) / 2
        exp_term_1 = np.exp(-self.hyper_para.k * grad_avg)
        exp_term_2 = np.exp(-self.hyper_para.k)
        self.pre_edgeweight = self.hyper_para.c * (exp_term_1 - exp_term_2)
        self.pre_edgeweight /= self.vox_num

    def log_vmf_normalization_constant(self, d, k):
        sizek = k.shape
        k = k.flatten()
        out = (d / 2 - 1) * np.log(k + 1e-13) - np.log(besseli((d / 2 - 1) * np.ones_like(k), k) + 1e-13)
        if d < 1000:
            k0 = 10
        elif 1000 < d < 2000:
            k0 = 500
        elif 2000 <= d < 5000:
            k0 = 1600
        elif 30000 <= d < 40000:
            k0 = 12500
        elif 40000 <= d < 50000:
            k0 = 14400
        elif 150000 <= d < 160000:
            k0 = 51200
        elif d >= 300000:
            k0 = 102288
        fk0 = (d / 2 - 1) * np.log(k0 + 1e-13) - np.log(besseli(d / 2 - 1, k0) + 1e-13)
        if np.isinf(fk0):
            k0 = 0.331 * d
            fk0 = (d / 2 - 1) * np.log(k0 + 1e-13) - np.log(besseli(d / 2 - 1, k0) + 1e-13)
        nGrids = 1000
        maskof = np.where(np.maximum(k > k0, np.isinf(out)))[0]
        nkof = len(maskof)
        if nkof > 0:
            print('setting a constant in computing Normalization log zn')
            kof = k[maskof]
            ofintv = (kof - k0) / nGrids
            tempcnt = np.arange(1, nGrids + 1) - 0.5
            ks = k0 + np.outer(kof, tempcnt) * ofintv[:, np.newaxis]
            CBIG_Adsum = np.sum(1 / (0.5 * (d - 1) / ks + np.sqrt(1 + (0.5 * (d - 1) / ks) ** 2)), axis=1)
            out[maskof] = fk0 - ofintv * CBIG_Adsum
        out = out.reshape(sizek)
        return out

    def loss1(self):
        unary = np.zeros((self.vox_num, self.roi_num))
        nlv_cons = -self.log_vmf_normalization_constant(self.time_len, self.hyper_para.kappas)
        nlv_cons = nlv_cons[1:] / self.vox_num
        nlv_cons = np.tile(nlv_cons, (self.vox_num, 1))
        unary[:, :] += nlv_cons
        return unary

    def loss2(self):
        unary = np.zeros((self.vox_num, self.roi_num))
        for roi_label in range(1, self.roi_num + 1):
            vox_roi_indexes = np.where(self.label == roi_label)[0]
            vox2roi_corre = self.pre_dot_acti_mat[vox_roi_indexes, :]
            numerators = vox2roi_corre.sum(axis=0)
            denominator = np.sqrt(np.sum(vox2roi_corre[:, vox_roi_indexes])) + 0.0001
            unary[:, roi_label - 1] -= self.hyper_para.kappas[roi_label] * numerators / denominator / self.vox_num
        return unary

    def loss3(self):
        unary = np.zeros((self.vox_num, self.roi_num))
        nlv_cons = -self.log_vmf_normalization_constant(3, self.hyper_para.taos)
        nlv_cons = nlv_cons[1:] / self.vox_num
        nlv_cons = np.tile(nlv_cons, (self.vox_num, 1))
        unary[:, :] += nlv_cons
        return unary

    def loss4(self):
        unary = np.zeros((self.vox_num, self.roi_num))
        for roi_label in range(1, self.roi_num + 1):
            vox_roi_indexes = np.where(self.label == roi_label)[0]
            vox2roi_corre = self.pre_dot_coord_mat[vox_roi_indexes, :]
            numerators = np.sum(vox2roi_corre, axis=0)
            denominator = np.sqrt(np.sum(vox2roi_corre[:, vox_roi_indexes])) + 0.0001
            unary[:, roi_label - 1] -= self.hyper_para.taos[roi_label] * numerators / denominator / self.vox_num
        return unary

    def unary_calcu(self):
        unary = np.zeros((self.vox_num, self.roi_num))
        unary += self.loss1()
        unary += self.loss2()
        unary += self.loss3()
        unary += self.loss4()
        unary -= unary.min()
        return unary

    def edgeweight_calcu(self):
        condition = self.label[self.edges[:, 0]] != self.label[self.edges[:, 1]]
        edge_weights = copy.deepcopy(self.pre_edgeweight)
        edge_weights[~condition] = 0
        return edge_weights

    def increase_element_count(self, arr):
        for (element, count) in zip(*np.unique(arr, return_counts=True)):
            if count < 100:
                indices = np.where(arr != element)[0].tolist()
                num_replace = 100 - count
                if len(indices) < num_replace:
                    replace_indices = random.choices(range(len(arr)), k=num_replace)
                else:
                    replace_indices = random.sample(indices, num_replace)
                arr[replace_indices] = element
        return arr

    def graphcut_opti(self, if_end):
        edge_weights = self.edgeweight_calcu().astype(np.float64)
        unary = self.unary_calcu().astype(np.float64)
        pairwise = (1 - np.eye(self.roi_num)).astype(np.float64)
        if hasattr(gco, 'cut_general_graph'):
            self.label = gco.cut_general_graph(self.edges, edge_weights, unary, pairwise, n_iter=-1, algorithm='swap')
        else:
            gc = gco.GCOGeneralGraph(self.vox_num, self.roi_num)
            gc.set_data_cost(unary)
            gc.set_smooth_cost(pairwise)
            for k in range(len(self.edges)):
                (i, j) = (int(self.edges[k, 0]), int(self.edges[k, 1]))
                if i < j:
                    gc.set_neighbors(i, j, float(edge_weights[k]))
            gc.swap(-1)
            self.label = np.array([gc.get_label(s) for s in range(self.vox_num)], dtype=np.int32)
        self.label = self.label + 1
        if not if_end:
            self.label[self.ref_index] = self.ref_template
        if if_end:
            full_set = set(range(1, self.roi_num + 1))
            labels_set = set(self.label)
            missing_rois = list(full_set - labels_set)
            missing_rois = [element for element in missing_rois for _ in range(1)]
            selected_indexs = random.sample(list(range(self.vox_num)), len(missing_rois))
            self.label[selected_indexs] = missing_rois

    def hyper_para_renew(self):
        M = self.time_len
        gammas = np.zeros((self.roi_num + 1,))
        for roi_label in range(1, self.roi_num + 1):
            vox_roi_indexes = np.where(self.label == roi_label)[0]
            vox2roi_corre = self.pre_dot_acti_mat[vox_roi_indexes, :]
            denominator = np.sqrt(np.sum(vox2roi_corre[:, vox_roi_indexes])) + 0.0001
            gammas[roi_label] += denominator
        gammas /= self.vox_num
        finite_arr = gammas[~np.isinf(gammas)]
        max_value = np.max(finite_arr)
        inf_mask = np.isinf(gammas)
        gammas[inf_mask] = max_value
        self.hyper_para.kappas = (M - 2) * gammas / (1 - gammas * gammas + 0.0001) + (M - 1) * gammas / (2 * (M - 2) + 0.0001)

    def run(self, num_iterations=30):
        len = num_iterations
        if_end = False
        for i in range(len):
            print('运行到了半脑第{0}轮'.format(i))
            t1 = time.time()
            if i == len - 1:
                if_end = True
            self.graphcut_opti(if_end)
            t2 = time.time()
            print('graphcut_opti: {0}'.format(t2 - t1))
            self.hyper_para_renew()
        parent_dir = os.path.dirname(os.path.dirname(self.dataset))
        dataset_name = os.path.basename(self.dataset)
        output_dir = os.path.join(parent_dir, 'Tmp_Res', dataset_name, 'MRF_Output_Enhanced')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, 'temp_label_enhanced.npy')
        np.save(output_path, self.label)
        print(f'\n✓ 脑分区标签已保存到: {output_path}')
        print(f'  标签形状: {self.label.shape}')
        print(f'  标签范围: [{np.min(self.label)}, {np.max(self.label)}]')
if __name__ == '__main__':
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datasets = [os.path.join(current_dir, 'tmp/test_dataset_lh')]
    for dataset in datasets:
        mrf2 = MarkovRandomFieldEnhanced(dataset)
        mrf2.run()
