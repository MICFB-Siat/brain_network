import sys
import os
current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)
import numpy as np
import nibabel as nib
from Util import Util
import shutil
import subprocess
from nibabel.cifti2 import Cifti2Image, Cifti2BrainModel
import gc
import math
from scipy.io import loadmat
import torch
import time
import argparse
from PathManager import PathManager
_path_manager = None

def get_pm():
    global _path_manager
    if _path_manager is None:
        _path_manager = PathManager()
    return _path_manager

def calculate_correlation_gpu(fmri_data, device='cuda', batch_size=5000):
    print(f'  开始GPU加速相关性计算 (设备: {device})...')
    start_time = time.time()
    (n_timepoints, n_vertices) = fmri_data.shape
    fmri_mean = fmri_data.mean(axis=0, keepdims=True)
    fmri_std = fmri_data.std(axis=0, keepdims=True)
    fmri_std[fmri_std == 0] = 1
    fmri_normalized = (fmri_data - fmri_mean) / fmri_std
    fmri_tensor = torch.from_numpy(fmri_normalized).float().to(device)
    correlation_matrix = torch.zeros((n_vertices, n_vertices), dtype=torch.float32, device=device)
    print(f'    分块计算相关性矩阵 (batch_size={batch_size})...')
    n_batches = (n_vertices + batch_size - 1) // batch_size
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_vertices)
        batch_corr = torch.mm(fmri_tensor[:, start_idx:end_idx].T, fmri_tensor) / (n_timepoints - 1)
        correlation_matrix[start_idx:end_idx, :] = batch_corr
        if (i + 1) % 10 == 0 or i + 1 == n_batches:
            print(f'      进度: {i + 1}/{n_batches} 批次完成')
    correlation_np = correlation_matrix.cpu().numpy()
    elapsed_time = time.time() - start_time
    print(f'  ✓ GPU相关性计算完成，耗时: {elapsed_time:.2f}秒')
    return correlation_np

def calculate_cifti_correlation(input_cifti, output_cifti, apply_fisher_z=False, use_covariance=False, mem_limit=None, use_gpu=True, gpu_id=0):
    if use_gpu and torch.cuda.is_available():
        print(f'  使用GPU加速计算相关性矩阵 (GPU {gpu_id})...')
        cifti_img = nib.load(input_cifti)
        fmri_data = cifti_img.get_fdata()
        device = f'cuda:{gpu_id}'
        correlation_matrix = calculate_correlation_gpu(fmri_data, device=device)
        if apply_fisher_z:
            print(f'  应用 Fisher z 变换...')
            correlation_matrix = np.clip(correlation_matrix, -0.9999, 0.9999)
            correlation_matrix = 0.5 * np.log((1 + correlation_matrix) / (1 - correlation_matrix))
        header = cifti_img.header
        brain_axis = header.get_axis(1)
        new_header = header.from_axes((brain_axis, brain_axis))
        output_img = Cifti2Image(dataobj=correlation_matrix, header=new_header)
        output_img.to_filename(output_cifti)
        print(f'  ✓ 相关性矩阵已保存到: {output_cifti}')
    else:
        print(f'  使用 wb_command 计算相关性矩阵（CPU模式）...')
        command = ['wb_command', '-cifti-correlation', input_cifti, output_cifti]
        if apply_fisher_z:
            command.append('-fisher-z')
        if use_covariance:
            command.append('-covariance')
        if mem_limit is not None:
            command.extend(['-mem-limit', str(mem_limit)])
        try:
            subprocess.run(command, check=True)
            print(f'  ✓ 相关性矩阵计算完成，结果保存到 {output_cifti}')
        except subprocess.CalledProcessError as e:
            print(f'  ✗ 执行命令时出错: {e}')

def rsfc_gradient_rsfc_gradient(corr_cifti_file, surface_file, gradient_file, hemi='L'):
    direction = 'ROW'
    if hemi == 'L':
        command = f'wb_command -cifti-gradient {corr_cifti_file} {direction} {gradient_file} -left-surface {surface_file}'
    elif hemi == 'R':
        command = f'wb_command -cifti-gradient {corr_cifti_file} {direction} {gradient_file} -right-surface {surface_file}'
    else:
        print('你的hemi参数输入出错啦啦啦啦啦啦啦啦啦啦！！！！！！！！！！！！！！！！！！！！！！！！！！')
    try:
        subprocess.run(command, shell=True, check=True)
        print('梯度矩阵计算完成。')
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f'  wb_command 不可用或执行失败，使用 Python 梯度代理: {e}')
        try:
            cifti_img = nib.load(corr_cifti_file)
            corr_mat = cifti_img.get_fdata()
            if corr_mat.ndim != 2 or corr_mat.shape[0] != corr_mat.shape[1]:
                raise ValueError('相关性文件应为 (V,V) 矩阵')
            grad_proxy = 1.0 - np.clip(corr_mat.astype(np.float64), -1, 1)
            header = cifti_img.header
            brain_axis = header.get_axis(1)
            new_header = header.from_axes((brain_axis, brain_axis))
            out_img = Cifti2Image(dataobj=grad_proxy.astype(np.float32), header=new_header)
            out_img.to_filename(gradient_file)
            print('  ✓ 已保存梯度代理 (1-correlation) 到: %s' % gradient_file)
        except Exception as e2:
            print(f'  Python 梯度代理失败: {e2}')
            raise

def compute_fmri_average(dataset_name, dataset_path):
    if dataset_name[-2:] == 'lh':
        expected_vertices = 29696
    elif dataset_name[-2:] == 'rh':
        expected_vertices = 29716
    else:
        raise ValueError(f"数据集名称必须以 '_lh' 或 '_rh' 结尾: {dataset_name}")
    tmp_dataset_path = os.path.join(current_dir_path, 'tmp', dataset_name)
    cached_fmri_path = os.path.join(tmp_dataset_path, 'fmri_avg.npy')
    if os.path.exists(cached_fmri_path):
        print(f'  ✓ 发现缓存的平均fMRI，直接加载: {cached_fmri_path}')
        fmri_avg = np.load(cached_fmri_path)
        print(f'    形状: {fmri_avg.shape}')
        if fmri_avg.shape[1] != expected_vertices:
            print(f'  ⚠️  警告: 缓存文件形状错误！')
            print(f'    预期: (time_points, {expected_vertices})')
            print(f'    实际: {fmri_avg.shape}')
            print(f'    删除损坏的缓存文件并重新计算...')
            os.remove(cached_fmri_path)
            return compute_fmri_average(dataset_name, dataset_path)
        fmri_files = [f for f in os.listdir(dataset_path) if f.endswith('.dtseries.nii')]
        return (fmri_avg, len(fmri_files), fmri_avg.shape[0], [])
    print(f'  缓存不存在，开始计算平均fMRI...')
    fmri_files = [f for f in os.listdir(dataset_path) if f.endswith('.dtseries.nii')]
    skipped_files = []
    print(f'  第一遍扫描：确定最短时间序列长度...')
    shortest_fmrilen = math.inf
    valid_files = []
    for (idx, filename) in enumerate(fmri_files, 1):
        fmri_path = os.path.join(dataset_path, filename)
        try:
            fmri = nib.load(fmri_path)
            time_length = fmri.shape[0]
            shortest_fmrilen = min(shortest_fmrilen, time_length)
            valid_files.append(filename)
            if idx % 10 == 0:
                print(f'    已扫描 {idx}/{len(fmri_files)} 个文件...')
        except (OSError, Exception) as e:
            print(f'  ⚠️  跳过损坏文件: {filename} (错误: {str(e)[:100]})')
            skipped_files.append(filename)
    if len(valid_files) == 0:
        raise ValueError(f'数据集 {dataset_name} 中没有找到有效的 .dtseries.nii 文件')
    print(f'  ✓ 找到 {len(valid_files)} 个有效文件，最短时间序列: {shortest_fmrilen} 时间点')
    print(f'  第二遍扫描：累积平均 fMRI 数据...')
    fmri_sum = None
    for (idx, filename) in enumerate(valid_files, 1):
        fmri_path = os.path.join(dataset_path, filename)
        try:
            fmri_data = nib.load(fmri_path).get_fdata()
            fmri_data = Util.norm_norm(fmri_data)
            fmri_data = fmri_data[:shortest_fmrilen, :]
            if fmri_data.shape[1] != expected_vertices:
                print(f'  ⚠️  警告: 文件 {filename} 顶点数不匹配!')
                print(f'    预期: {expected_vertices}, 实际: {fmri_data.shape[1]}')
                print(f'    跳过该文件...')
                continue
            if fmri_sum is None:
                fmri_sum = fmri_data.astype(np.float64)
            else:
                fmri_sum += fmri_data
            if idx % 10 == 0:
                print(f'    已处理 {idx}/{len(valid_files)} 个被试...')
        except Exception as e:
            print(f'  ⚠️  处理文件 {filename} 时出错: {e}')
            continue
    fmri_avg = fmri_sum / len(valid_files)
    del fmri_sum
    gc.collect()
    if fmri_avg.shape[1] != expected_vertices:
        raise ValueError(f'❌ 计算出的平均 fMRI 形状错误！\n  数据集: {dataset_name}\n  预期形状: (time_points, {expected_vertices})\n  实际形状: {fmri_avg.shape}\n  这可能是由于数据集中包含不同半球的数据导致。\n')
    os.makedirs(tmp_dataset_path, exist_ok=True)
    np.save(cached_fmri_path, fmri_avg.astype(np.float32))
    print(f'  ✓ 保存平均fMRI到缓存: {cached_fmri_path}')
    print(f'  ✓ 完成平均计算：{len(valid_files)} 个被试')
    return (fmri_avg, len(valid_files), shortest_fmrilen, skipped_files)

def pre_multi_acti(dataset_name, dataset_path, fmri_avg_cached=None, use_gpu=True, gpu_id=0):
    if dataset_name[-2:] == 'lh':
        expected_vertices = 29696
    elif dataset_name[-2:] == 'rh':
        expected_vertices = 29716
    else:
        raise ValueError(f"数据集名称必须以 '_lh' 或 '_rh' 结尾: {dataset_name}")
    if fmri_avg_cached is not None:
        print(f'  使用缓存的平均 fMRI 数据')
        fmri_avg = fmri_avg_cached
        shortest_fmrilen = fmri_avg.shape[0]
        fmri_count = 'cached'
        skipped_files = []
    else:
        print(f'  正在加载并平均所有 fMRI 文件...')
        (fmri_avg, fmri_count, shortest_fmrilen, skipped_files) = compute_fmri_average(dataset_name, dataset_path)
    if fmri_avg.shape[1] != expected_vertices:
        tmp_dataset_path = os.path.join(current_dir_path, 'tmp', dataset_name)
        cached_fmri_path = os.path.join(tmp_dataset_path, 'fmri_avg.npy')
        raise ValueError(f'❌ fMRI 平均数据形状错误！\n  数据集: {dataset_name}\n  预期形状: (time_points, {expected_vertices})\n  实际形状: {fmri_avg.shape}\n  \n  这通常是由于缓存文件损坏导致。解决方法：\n  1. 删除缓存目录: rm -rf {tmp_dataset_path}\n  2. 重新运行 MRF 准备步骤\n')
    print(f'  ✓ 形状验证通过: {fmri_avg.shape}')
    print(f'  计算平均 fMRI 的预乘矩阵 (时间长度: {shortest_fmrilen})...')
    if use_gpu and torch.cuda.is_available():
        print(f'  使用GPU加速矩阵乘法 (GPU {gpu_id})...')
        start_time = time.time()
        device = f'cuda:{gpu_id}'
        fmri_tensor = torch.from_numpy(fmri_avg).float().to(device)
        pre_acti_multi_tensor = torch.mm(fmri_tensor.T, fmri_tensor)
        pre_acti_multi = pre_acti_multi_tensor.cpu().numpy()
        del fmri_tensor, pre_acti_multi_tensor
        torch.cuda.empty_cache()
        elapsed_time = time.time() - start_time
        print(f'  ✓ GPU矩阵乘法完成，耗时: {elapsed_time:.2f}秒')
    else:
        print(f'  使用CPU计算矩阵乘法...')
        start_time = time.time()
        fmri_avg_fp32 = fmri_avg.astype(np.float32)
        pre_acti_multi = np.dot(fmri_avg_fp32.T, fmri_avg_fp32)
        del fmri_avg_fp32
        elapsed_time = time.time() - start_time
        print(f'  ✓ CPU矩阵乘法完成，耗时: {elapsed_time:.2f}秒')
    output_dir = os.path.join(dataset_path, 'MRF_Prepare')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, dataset_name[-2:] + '_pre_dot_acti_mat.npy')
    np.save(output_path, pre_acti_multi)
    gc.collect()
    print(f'  ✓ 保存活动预乘矩阵到: {output_path}')
    print(f'  成功处理: {fmri_count} 个文件, 跳过: {len(skipped_files)} 个损坏文件')
    print(f'  最短时间序列长度: {shortest_fmrilen} 时间点')
    if skipped_files:
        print(f"  跳过的文件: {', '.join(skipped_files[:5])}" + (' ...' if len(skipped_files) > 5 else ''))
    return fmri_avg

def pre_grad(dataset_name, dataset_path, fmri_avg_cached=None, use_gpu=True, gpu_id=0, compute_gradient=True):
    output_dir = os.path.join(dataset_path, 'MRF_Prepare')
    os.makedirs(output_dir, exist_ok=True)
    if dataset_name[-2:] == 'lh':
        hemi = 'lh'
        shared_gradient_file = os.path.join(current_dir_path, 'tmp', 'lh_gradient_matrix.dscalar.nii')
        output_gradient_file = os.path.join(output_dir, 'lh_gradient_matrix.dscalar.nii')
    elif dataset_name[-2:] == 'rh':
        hemi = 'rh'
        shared_gradient_file = os.path.join(current_dir_path, 'tmp', 'rh_gradient_matrix.dscalar.nii')
        output_gradient_file = os.path.join(output_dir, 'rh_gradient_matrix.dscalar.nii')
    else:
        raise ValueError(f"数据集名称必须以 '_lh' 或 '_rh' 结尾: {dataset_name}")
    if os.path.exists(shared_gradient_file):
        print(f'  发现已有的共享梯度文件，直接复制: {shared_gradient_file}')
        shutil.copy(shared_gradient_file, output_gradient_file)
        print(f'  ✓ 复制梯度矩阵到: {output_gradient_file}')
        return
    if not compute_gradient:
        print(f'  ⊘ 共享梯度文件不存在且已跳过梯度计算（根据 --skip-gradient 参数）')
        print(f'    警告: 没有可用的梯度文件用于 {dataset_name}！')
        return
    print(f'  共享梯度文件不存在，开始计算梯度（这可能需要较长时间）...')
    if fmri_avg_cached is not None:
        print(f'  使用缓存的平均 fMRI 数据')
        fmri_avg = fmri_avg_cached
        fmri_count = 'cached'
        skipped_files = []
    else:
        print(f'  正在加载并平均所有 fMRI 文件...')
        (fmri_avg, fmri_count, shortest_fmrilen, skipped_files) = compute_fmri_average(dataset_name, dataset_path)
    medial_mask_path = os.path.join(current_dir_path, 'tmp', 'fs_LR_32k_medial_mask.mat')
    tmp_shared_dir = os.path.join(current_dir_path, 'tmp')
    if hemi == 'lh':
        medial = np.array(loadmat(medial_mask_path)['medial_mask'])
        medial = medial[:medial.shape[0] // 2, 0]
        template_header_path = os.path.join(tmp_shared_dir, 'lh_rsfmri.dtseries.nii')
        if not os.path.exists(template_header_path):
            raise FileNotFoundError(f'模板 header 文件不存在: {template_header_path}')
        fmri_head = nib.load(template_header_path).header
    else:
        medial = np.array(loadmat(medial_mask_path)['medial_mask'])
        medial = medial[medial.shape[0] // 2:, 0]
        template_header_path = os.path.join(tmp_shared_dir, 'rh_rsfmri.dtseries.nii')
        if not os.path.exists(template_header_path):
            raise FileNotFoundError(f'模板 header 文件不存在: {template_header_path}')
        fmri_head = nib.load(template_header_path).header
    non_medial_vertices = int(medial.sum())
    total_vertices = medial.shape[0]
    template_brain_dim = nib.load(template_header_path).shape[1]
    if fmri_avg.shape[1] == non_medial_vertices:
        if template_brain_dim == non_medial_vertices:
            print(f'  检测到非medial格式数据 ({fmri_avg.shape[1]}顶点)，模板一致，直接使用')
            fmri_head.matrix._mims[0].number_of_series_points = fmri_avg.shape[0]
            fmri_cifti = Cifti2Image(dataobj=fmri_avg, header=fmri_head)
        else:
            print(f'  检测到非medial格式数据 ({fmri_avg.shape[1]}顶点)，还原到完整格式 ({total_vertices}顶点)')
            fmri_avg_full = np.zeros((fmri_avg.shape[0], total_vertices), dtype=fmri_avg.dtype)
            fmri_avg_full[:, np.where(medial == 1)[0]] = fmri_avg
            fmri_head.matrix._mims[0].number_of_series_points = fmri_avg_full.shape[0]
            fmri_cifti = Cifti2Image(dataobj=fmri_avg_full, header=fmri_head)
    elif fmri_avg.shape[1] == total_vertices:
        print(f'  检测到完整格式数据 ({total_vertices}顶点)，直接使用')
        fmri_head.matrix._mims[0].number_of_series_points = fmri_avg.shape[0]
        fmri_cifti = Cifti2Image(dataobj=fmri_avg, header=fmri_head)
    else:
        raise ValueError(f'fMRI数据顶点数({fmri_avg.shape[1]})与预期不符: 非medial顶点数={non_medial_vertices}, 完整顶点数={total_vertices}')
    cifti_file = os.path.join(output_dir, 'temp_fmri.dtseries.nii')
    fmri_cifti.to_filename(cifti_file)
    input_cifti_file = cifti_file
    output_cifti_file = os.path.join(output_dir, 'correlation_output.dscalar.nii')
    calculate_cifti_correlation(input_cifti_file, output_cifti_file, apply_fisher_z=False, use_covariance=False, mem_limit=2, use_gpu=use_gpu, gpu_id=gpu_id)
    corr_cifti_file = output_cifti_file
    surface_base_path = os.path.join(current_dir_path, 'tmp')
    if hemi == 'lh':
        surface_file = os.path.join(surface_base_path, 'S1200.L.sphere.32k_fs_LR.surf.gii')
        rsfc_gradient_rsfc_gradient(corr_cifti_file, surface_file, output_gradient_file, hemi='L')
    else:
        surface_file = os.path.join(surface_base_path, 'S1200.R.sphere.32k_fs_LR.surf.gii')
        rsfc_gradient_rsfc_gradient(corr_cifti_file, surface_file, output_gradient_file, hemi='R')
    os.remove(cifti_file)
    os.remove(output_cifti_file)
    print(f'  ✓ 保存梯度矩阵到: {output_gradient_file}')
    print(f'  成功处理: {fmri_count} 个文件, 跳过: {len(skipped_files)} 个损坏文件')
    if skipped_files:
        print(f"  跳过的文件: {', '.join(skipped_files[:5])}" + (' ...' if len(skipped_files) > 5 else ''))
    shutil.copy(output_gradient_file, shared_gradient_file)
    print(f'  ✓ 保存共享梯度文件到: {shared_gradient_file} (供后续数据集重用)')

def _generate_edges_and_coord_from_surface(tmp_shared_dir, hemi_prefix):
    surface_path = os.path.join(tmp_shared_dir, 'S1200.L.sphere.32k_fs_LR.surf.gii' if hemi_prefix == 'lh' else 'S1200.R.sphere.32k_fs_LR.surf.gii')
    medial_path = os.path.join(tmp_shared_dir, 'fs_LR_32k_medial_mask.mat')
    if not os.path.isfile(surface_path) or not os.path.isfile(medial_path):
        return False
    medial = np.array(loadmat(medial_path)['medial_mask'])
    hemi_medial = medial[:medial.shape[0] // 2, 0] if hemi_prefix == 'lh' else medial[medial.shape[0] // 2:, 0]
    non_medial_idx = np.where(hemi_medial == 1)[0]
    n_non = len(non_medial_idx)
    medial2non = np.full(hemi_medial.shape[0], -1, dtype=np.int32)
    medial2non[non_medial_idx] = np.arange(n_non)
    gii = nib.load(surface_path)
    verts = gii.darrays[0].data
    tris = gii.darrays[1].data
    if verts.shape[0] != hemi_medial.shape[0]:
        return False
    coord_non = verts[non_medial_idx]
    coord_non = coord_non.T
    norms = np.linalg.norm(coord_non, axis=0)
    norms[norms == 0] = 1
    coord_non = coord_non / norms[np.newaxis, :]
    pre_dot_coord = (coord_non.T.astype(np.float64) @ coord_non.astype(np.float64)).astype(np.float32)
    edges_set = set()
    for t in tris:
        (a, b, c) = (int(t[0]), int(t[1]), int(t[2]))
        if medial2non[a] < 0 or medial2non[b] < 0 or medial2non[c] < 0:
            continue
        (na, nb, nc) = (medial2non[a], medial2non[b], medial2non[c])
        for (u, v) in [(na, nb), (nb, nc), (nc, na)]:
            if u > v:
                (u, v) = (v, u)
            edges_set.add((u, v))
    edges = np.array(sorted(edges_set), dtype=np.int32)
    if edges.size == 0:
        edges = np.zeros((0, 2), dtype=np.int32)
    coord_path = os.path.join(tmp_shared_dir, hemi_prefix + '_pre_dot_coord_mat.npy')
    edges_path = os.path.join(tmp_shared_dir, hemi_prefix + '_edges.npy')
    np.save(coord_path, pre_dot_coord)
    np.save(edges_path, edges)
    print(f'  ✓ 已从表面生成: {coord_path} (shape {pre_dot_coord.shape}), {edges_path} (shape {edges.shape})')
    return True

def pre_multi_coord_and_edges_and_medial_and_index(dataset_name, dataset_path=None):
    tmp_shared_dir = os.path.join(current_dir_path, 'tmp')
    if dataset_path:
        tmp_dataset_dir = dataset_path
    else:
        pm = get_pm()
        tmp_dataset_dir = pm.get_tmp_path(dataset_name, create=False)
    if dataset_name[-2:] == 'lh':
        src_coord_path = os.path.join(tmp_shared_dir, 'lh_pre_dot_coord_mat.npy')
        src_edges_path = os.path.join(tmp_shared_dir, 'lh_edges.npy')
    elif dataset_name[-2:] == 'rh':
        src_coord_path = os.path.join(tmp_shared_dir, 'rh_pre_dot_coord_mat.npy')
        src_edges_path = os.path.join(tmp_shared_dir, 'rh_edges.npy')
    else:
        raise ValueError(f"数据集名称必须以 '_lh' 或 '_rh' 结尾: {dataset_name}")
    src_medial_path = os.path.join(tmp_shared_dir, 'fs_LR_32k_medial_mask.mat')
    src_index_path = os.path.join(tmp_dataset_dir, 'FeatExt_Prepare', 'index.npy')
    output_dir = os.path.join(tmp_dataset_dir, 'MRF_Prepare')
    os.makedirs(output_dir, exist_ok=True)
    dst_coord_path = os.path.join(output_dir, os.path.basename(src_coord_path))
    dst_edges_path = os.path.join(output_dir, os.path.basename(src_edges_path))
    dst_medial_path = os.path.join(output_dir, os.path.basename(src_medial_path))
    dst_index_path = os.path.join(output_dir, os.path.basename(src_index_path))
    hemi_prefix = 'lh' if dataset_name.endswith('_lh') else 'rh'
    need_gen = not os.path.exists(src_coord_path) or not os.path.exists(src_edges_path)
    if os.path.exists(src_coord_path):
        try:
            sh = np.load(src_coord_path, mmap_mode='r').shape
            if len(sh) != 2 or sh[0] != sh[1]:
                need_gen = True
        except Exception:
            need_gen = True
    if need_gen:
        _generate_edges_and_coord_from_surface(tmp_shared_dir, hemi_prefix)
    if os.path.exists(src_coord_path):
        shutil.copy(src_coord_path, dst_coord_path)
        print(f'  ✓ 复制共享坐标矩阵: {dst_coord_path}')
    else:
        print(f'  ⚠ 警告: 共享坐标文件不存在: {src_coord_path}')
        print(f'     请确保已在 ./tmp/ 目录中准备好共享的几何文件')
    if os.path.exists(src_edges_path):
        shutil.copy(src_edges_path, dst_edges_path)
        print(f'  ✓ 复制共享边文件: {dst_edges_path}')
    else:
        print(f'  ⚠ 警告: 共享边文件不存在: {src_edges_path}')
        print(f'     请确保已在 ./tmp/ 目录中准备好共享的几何文件')
    if os.path.exists(src_medial_path):
        shutil.copy(src_medial_path, dst_medial_path)
        print(f'  ✓ 复制 medial mask: {dst_medial_path}')
    else:
        print(f'  ⚠ 警告: 源文件不存在: {src_medial_path}')
    if os.path.exists(src_index_path):
        shutil.copy(src_index_path, dst_index_path)
        print(f'  ✓ 复制索引文件: {dst_index_path}')
    else:
        print(f'  ⚠ 警告: 源文件不存在: {src_index_path}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MRF准备脚本 - 支持GPU加速和可选梯度计算', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\n示例用法:\n  # 默认模式（GPU加速 + 计算梯度）\n  python mrf_prepare.py\n\n  # 跳过梯度计算（节省时间）\n  python mrf_prepare.py --skip-gradient\n\n  # 使用CPU模式\n  python mrf_prepare.py --no-gpu\n\n  # 指定GPU设备\n  python mrf_prepare.py --gpu-id 3\n\n  # 组合使用\n  python mrf_prepare.py --skip-gradient --gpu-id 5\n        ')
    parser.add_argument('--skip-gradient', action='store_true', help='跳过梯度文件计算（不计算相关性矩阵和梯度，节省时间）')
    parser.add_argument('--no-gpu', action='store_true', help='禁用GPU加速，使用CPU模式')
    parser.add_argument('--gpu-id', type=int, default=0, help='指定GPU设备ID（默认: 0）')
    args = parser.parse_args()
    USE_GPU = not args.no_gpu
    GPU_ID = args.gpu_id
    COMPUTE_GRADIENT = not args.skip_gradient
    datasets = [('test_dataset_lh', os.path.join(current_dir_path, 'tmp/test_dataset_lh'))]
    print('=' * 70)
    print('MRF 准备脚本 (GPU加速版)')
    print('=' * 70)
    print(f"GPU加速: {('启用' if USE_GPU else '禁用')}")
    if USE_GPU:
        print(f'GPU设备: cuda:{GPU_ID}')
        if torch.cuda.is_available():
            print(f'GPU型号: {torch.cuda.get_device_name(GPU_ID)}')
            print(f'显存容量: {torch.cuda.get_device_properties(GPU_ID).total_memory / 1024 ** 3:.1f} GB')
        else:
            print('⚠️  警告: CUDA不可用，将回退到CPU模式')
            USE_GPU = False
    print(f"梯度计算: {('启用' if COMPUTE_GRADIENT else '禁用 (--skip-gradient)')}")
    print('\n注意: 请在运行前修改 __main__ 部分的数据集路径配置!\n')
    for (dataset_name, dataset_path) in datasets:
        if not os.path.exists(dataset_path):
            print(f'\n警告: 数据集路径不存在，跳过 {dataset_name}')
            print(f'  路径: {dataset_path}')
            continue
        print(f'\n处理数据集: {dataset_name}')
        print(f'  路径: {dataset_path}')
        output_dir = os.path.join(current_dir_path, 'tmp', dataset_name, 'MRF_Prepare')
        os.makedirs(output_dir, exist_ok=True)
        try:
            print('  步骤 1/3: 准备活动预乘矩阵...')
            fmri_avg = pre_multi_acti(dataset_name, dataset_path, use_gpu=USE_GPU, gpu_id=GPU_ID)
            print('  步骤 2/3: 准备梯度文件...')
            pre_grad(dataset_name, dataset_path, fmri_avg_cached=fmri_avg, use_gpu=USE_GPU, gpu_id=GPU_ID, compute_gradient=COMPUTE_GRADIENT)
            del fmri_avg
            gc.collect()
            print('  步骤 3/3: 复制坐标、边、medial mask 和索引文件...')
            pre_multi_coord_and_edges_and_medial_and_index(dataset_name)
            print(f'  ✓ 完成数据集 {dataset_name} 的 MRF 准备')
        except Exception as e:
            print(f'  ✗ 处理数据集 {dataset_name} 时出错: {e}')
            import traceback
            traceback.print_exc()
    print('\n' + '=' * 70)
    print('MRF 准备完成')
    print('=' * 70)
