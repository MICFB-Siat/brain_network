import os
import re
from typing import Dict, Optional, Sequence
import numpy as np
from nibabel import Cifti2Header, Cifti2Image
from nibabel.cifti2.cifti2 import Cifti2Label
from nilearn.image import load_img
from scipy.io import loadmat

def get_sequential_name(ref_name: str, counter_dict: Dict[str, int]) -> str:
    base_name = re.sub('_\\d+$', '', ref_name)
    if base_name not in counter_dict:
        counter_dict[base_name] = 0
    counter_dict[base_name] += 1
    return f'{base_name}_{counter_dict[base_name]}'

def _write_minimal_dlabel(class_labels: np.ndarray, output_path: str, stage4_dir: str) -> None:
    import nibabel as nib
    from nibabel.cifti2 import Cifti2Header, Cifti2Image
    from nibabel.cifti2.cifti2_axes import ScalarAxis
    tmp = os.path.join(stage4_dir, 'tmp')
    lh_path = os.path.join(tmp, 'lh_rsfmri.dtseries.nii')
    rh_path = os.path.join(tmp, 'rh_rsfmri.dtseries.nii')
    if not os.path.isfile(lh_path) or not os.path.isfile(rh_path):
        raise FileNotFoundError(f'Minimal dlabel requires tmp/lh_rsfmri.dtseries.nii and tmp/rh_rsfmri.dtseries.nii in {stage4_dir}')
    lh_img = nib.load(lh_path)
    rh_img = nib.load(rh_path)
    brain_lh = lh_img.header.get_axis(1)
    brain_rh = rh_img.header.get_axis(1)
    brain_combined = brain_lh + brain_rh
    scalar_axis = ScalarAxis(['Parcellation'])
    new_header = Cifti2Header.from_axes((scalar_axis, brain_combined))
    data = np.asarray(class_labels, dtype=np.float32).reshape(1, -1)
    img = Cifti2Image(data, new_header)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.to_filename(output_path)
    print(f'Saved minimal dlabel (no Schaefer template) to: {output_path}')

def generate_dlabel_from_labels(class_labels: np.ndarray, output_path: str, stage4_dir: str) -> None:
    network_counter: Dict[str, int] = {}
    medial_mask_path = os.path.join(stage4_dir, 'tmp', 'fs_LR_32k_medial_mask.mat')
    mat_contents = loadmat(medial_mask_path)
    mat_contents = np.array(mat_contents['medial_mask'])
    ref_template_path = os.path.join(stage4_dir, 'tmp', 'template', 'Schaefer2018_200Parcels_17Networks_order.dlabel.nii')
    try:
        ref_template = load_img(ref_template_path)
    except (FileNotFoundError, OSError) as e:
        print(f'  Schaefer template not found ({ref_template_path}), writing minimal dlabel.')
        lh_labels = class_labels[:29696].copy()
        rh_labels = class_labels[29696:].copy() + 100
        lh_labels[lh_labels == 0] = 1
        rh_labels[rh_labels == 100] = 101
        label_cat = np.concatenate((lh_labels, rh_labels))
        _write_minimal_dlabel(label_cat, output_path, stage4_dir)
        return
    ref_template_header: Cifti2Header = ref_template.header
    new_ref_header = ref_template_header.copy()
    ref_labels = ref_template.get_fdata()
    ref_labels = ref_labels[:, np.where(mat_contents[:, 0] == 1)[0]]
    ref_lh_label = ref_labels[0, :29696]
    ref_rh_label = ref_labels[0, 29696:]
    lh_labels = class_labels[:29696].copy()
    rh_labels = class_labels[29696:].copy() + 100
    lh_labels[lh_labels == 0] = 1
    rh_labels[rh_labels == 100] = 101
    print(f'ref_lh_label max: {np.max(ref_lh_label)}   ref_lh_label min: {np.min(ref_lh_label)}')
    print(f'ref_rh_labels max: {np.max(ref_rh_label)}   ref_rh_labels min: {np.min(ref_rh_label)}')
    print(f'lh_labels max: {np.max(lh_labels)}   lh_labels min: {np.min(lh_labels)}')
    print(f'rh_labels max: {np.max(rh_labels)}   rh_labels min: {np.min(rh_labels)}')
    ref_color_table = {}
    ref_name_table = {}
    maps = ref_template_header.get_index_map(0)
    for named_map in maps.named_maps:
        label_table = named_map.label_table
        if label_table is None:
            continue
        for label_key in label_table.keys():
            ref_color_table[label_key] = label_table[label_key].rgba
            ref_name_table[label_key] = label_table[label_key].label
    new_ref_map = new_ref_header.get_index_map(0)
    for new_ref_named_map in new_ref_map.named_maps:
        new_ref_label_table = new_ref_named_map.label_table
        if new_ref_label_table is None:
            continue
        keys_to_remove = [key for key in new_ref_label_table if key != 0]
        for key in keys_to_remove:
            del new_ref_label_table[key]
    for label in np.unique(lh_labels):
        try:
            label_index = np.where(lh_labels == label)
            (unique_elements, counts) = np.unique(ref_lh_label[label_index], return_counts=True)
            hash_stats = dict(zip(unique_elements, counts))
            max_key = max(hash_stats, key=hash_stats.get)
            if max_key == 0:
                max_key = 2
            label_name = get_sequential_name(ref_name_table[max_key], network_counter)
            print(label, label_name, ref_color_table[max_key])
            cifti2label = Cifti2Label(label, label_name, *ref_color_table[max_key])
            new_ref_map = new_ref_header.get_index_map(0)
            for new_ref_named_map in new_ref_map.named_maps:
                new_ref_label_table = new_ref_named_map.label_table
                new_ref_label_table[label] = cifti2label
                new_ref_header.get_axis(0)[label] = (ref_name_table[max_key], ref_color_table[max_key])
        except Exception:
            continue
    for label in np.unique(rh_labels):
        try:
            label_index = np.where(rh_labels == label)
            (unique_elements, counts) = np.unique(ref_rh_label[label_index], return_counts=True)
            hash_stats = dict(zip(unique_elements, counts))
            max_key = max(hash_stats, key=hash_stats.get)
            if max_key == 0:
                max_key = 102
            label_name = get_sequential_name(ref_name_table[max_key], network_counter)
            print(label, label_name, ref_color_table[max_key])
            cifti2label = Cifti2Label(label, label_name, *ref_color_table[max_key])
            new_ref_map = new_ref_header.get_index_map(0)
            for new_ref_named_map in new_ref_map.named_maps:
                new_ref_label_table = new_ref_named_map.label_table
                new_ref_label_table[label] = cifti2label
                new_ref_header.get_axis(0)[label] = (ref_name_table[max_key], ref_color_table[max_key])
        except Exception:
            continue
    class_labels_full = np.concatenate((lh_labels, rh_labels), axis=0)
    sdtemplate_list = [0] * len(mat_contents)
    i = 0
    j = 0
    while i < len(mat_contents):
        if mat_contents[i, 0] != 0:
            sdtemplate_list[i] = class_labels_full[j]
            j += 1
        i += 1
    sdtemplate_arr = np.array([sdtemplate_list])
    sdtemplate_img = Cifti2Image(sdtemplate_arr, new_ref_header)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sdtemplate_img.to_filename(output_path)
    print(f'Saved dlabel to: {output_path}')

def generate_dlabel_for_dataset_base(dataset_base: str, stage4_dir: str) -> str:
    lh_dataset = f'{dataset_base}_lh'
    rh_dataset = f'{dataset_base}_rh'
    lh_label_path = os.path.join(stage4_dir, 'Tmp_Res', lh_dataset, 'MRF_Output_Enhanced', 'temp_label_enhanced.npy')
    rh_label_path = os.path.join(stage4_dir, 'Tmp_Res', rh_dataset, 'MRF_Output_Enhanced', 'temp_label_enhanced.npy')
    if not os.path.exists(lh_label_path) or not os.path.exists(rh_label_path):
        raise FileNotFoundError(f"Missing LH/RH labels for base '{dataset_base}'.\n  LH: {lh_label_path}\n  RH: {rh_label_path}")
    lh_label = np.load(lh_label_path)
    rh_label = np.load(rh_label_path)
    label_cat = np.concatenate((lh_label, rh_label))
    output_dir = os.path.join(stage4_dir, 'Tmp_Res', dataset_base, 'MRF_Output_Enhanced')
    output_path = os.path.join(output_dir, 'SdTemplate.dlabel.nii')
    print(f'\nGenerating CIFTI visualization for {dataset_base}')
    print(f'  LH labels: {lh_label.shape}')
    print(f'  RH labels: {rh_label.shape}')
    print(f'  Output: {output_path}')
    generate_dlabel_from_labels(label_cat, output_path, stage4_dir)
    return output_path

def main(argv: Optional[Sequence[str]]=None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Generate SdTemplate.dlabel.nii for PFBN/Stage4 outputs (whole-brain).')
    parser.add_argument('--dataset-base', required=True, help='Base dataset name without hemisphere suffix, e.g. HCP_7T_RFMRI_Standard')
    parser.add_argument('--stage4-dir', default='../Stage4_Construct_Personalized_Functional_Brain_Network', help='Path to Stage4_Construct_Personalized_Functional_Brain_Network directory.')
    args = parser.parse_args(argv)
    stage4_dir = os.path.abspath(args.stage4_dir)
    if not os.path.isdir(stage4_dir):
        raise NotADirectoryError(f'stage4-dir not found: {stage4_dir}')
    generate_dlabel_for_dataset_base(args.dataset_base, stage4_dir)
if __name__ == '__main__':
    main()
