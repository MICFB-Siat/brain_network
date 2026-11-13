import os
import nibabel as nib
from nibabel.cifti2 import Cifti2Image
import numpy as np
from Util import Util

class HemisphereSplitter:
    LEFT_CORTEX_VERTICES = 29696
    RIGHT_CORTEX_VERTICES = 29716
    CORTEX_TOTAL_VERTICES = 59412

    def __init__(self, input_fmri_path):
        if not os.path.exists(input_fmri_path):
            raise FileNotFoundError(f'输入 fMRI 文件不存在: {input_fmri_path}')
        if not input_fmri_path.endswith('.dtseries.nii'):
            raise ValueError(f'输入文件必须是 .dtseries.nii 格式: {input_fmri_path}')
        self.input_path = input_fmri_path
        self.fmri_img = nib.load(input_fmri_path)
        self.fmri_data = self.fmri_img.get_fdata(dtype=np.float32)
        self.fmri_header = self.fmri_img.header
        print('对 fMRI 数据进行 L2 归一化（沿时间维度）...')
        self.fmri_data = Util.norm_norm(self.fmri_data)
        if self.fmri_data.shape[1] < self.CORTEX_TOTAL_VERTICES:
            raise ValueError(f'fMRI 数据维度不正确。期望至少 {self.CORTEX_TOTAL_VERTICES} 个顶点，实际得到 {self.fmri_data.shape[1]} 个顶点')
        self._split_data()
        (self.lh_header, self.rh_header) = self._construct_hemisphere_headers()

    def _split_data(self):
        cortex_data = self.fmri_data[:, :self.CORTEX_TOTAL_VERTICES]
        if self.fmri_data.shape[1] > self.CORTEX_TOTAL_VERTICES:
            print(f'⚠️  检测到非皮层数据，提取前 {self.CORTEX_TOTAL_VERTICES} 个顶点作为皮层')
        self.lh_data = cortex_data[:, :self.LEFT_CORTEX_VERTICES]
        self.rh_data = cortex_data[:, self.LEFT_CORTEX_VERTICES:]
        print(f'数据分割完成: 左 {self.lh_data.shape}, 右 {self.rh_data.shape}')

    def _construct_hemisphere_headers(self):
        from nibabel.cifti2.cifti2_axes import SeriesAxis, to_header
        n_time = self.lh_data.shape[0]
        time_axis = SeriesAxis(0.0, 1.0, n_time)
        orig_brain_axis = self.fmri_header.get_axis(1)
        lh_brain_model = rh_brain_model = None
        for (name, slc, bm) in orig_brain_axis.iter_structures():
            if 'CORTEX_LEFT' in str(name):
                lh_brain_model = bm
            elif 'CORTEX_RIGHT' in str(name):
                rh_brain_model = bm
        if lh_brain_model is None or rh_brain_model is None:
            raise ValueError('无法从原始数据中提取左/右半球脑模型')
        return (to_header((time_axis, lh_brain_model)), to_header((time_axis, rh_brain_model)))

    def save_hemispheres(self, output_dir=None, prefix=None):
        if output_dir is None:
            output_dir = os.path.dirname(self.input_path)
        else:
            os.makedirs(output_dir, exist_ok=True)
        if prefix is None:
            input_filename = os.path.basename(self.input_path)
            prefix = os.path.splitext(input_filename)[0]
            if prefix.endswith('.dtseries'):
                prefix = prefix[:-9]
        lh_path = os.path.join(output_dir, f'lh_{prefix}.dtseries.nii')
        rh_path = os.path.join(output_dir, f'rh_{prefix}.dtseries.nii')
        Cifti2Image(self.lh_data, self.lh_header).to_filename(lh_path)
        Cifti2Image(self.rh_data, self.rh_header).to_filename(rh_path)
        print(f'左半球已保存: {lh_path}\n右半球已保存: {rh_path}')
        return (lh_path, rh_path)
