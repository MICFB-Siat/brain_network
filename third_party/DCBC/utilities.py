import os, warnings, scipy
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
try:
    import torch as pt
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
warnings.filterwarnings('ignore', category=RuntimeWarning)

def sub2ind(array_shape, rows, cols):
    ind = rows * array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0] * array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0] * array_shape[1]] = -1
    rows = ind.astype('int') / array_shape[1]
    cols = ind % array_shape[1]
    return (rows, cols)

def scan_subdirs(path):
    sub_dirs = []
    for entry in os.scandir(path):
        if not entry.name.startswith('.') and entry.is_dir():
            sub_dirs.append(entry.name)
    return sub_dirs

def euclidean_distance(a, b, decimals=3):
    p1 = np.einsum('ij,ij->i', a, a)[:, np.newaxis]
    p2 = np.einsum('ij,ij->i', b, b)[:, np.newaxis]
    p3 = -2 * np.dot(a, b.T)
    dist = np.round(np.sqrt(p1 + p2 + p3), decimals)
    dist.flat[::dist.shape[0] + 1] = 0.0
    return dist

def compute_dist_from_surface(files, type, max_dist=50, hems='L', sparse=True):
    dist = []
    if files is None:
        file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parcellations', 'fs_LR_32k_template', 'fs_LR.32k.%s.sphere.surf.gii' % hems)
    else:
        file_name = files
    if type == 'euclidean':
        mat = nb.load(file_name)
        surf = [x.data for x in mat.darrays]
        surf_vertices = surf[0]
        dist = euclidean_distance(surf_vertices, surf_vertices)
        dist[dist > max_dist] = 0
    elif type == 'dijstra':
        mat = nb.load(file_name)
        surf = [x.data for x in mat.darrays]
        surf_vertices = surf[0]
    return scipy.sparse.csr_matrix(dist) if sparse else dist

def convert_numpy_to_torch_sparse(dist, device='cpu'):
    coo_matrix = dist.tocoo()
    indices = pt.LongTensor(np.vstack((coo_matrix.row, coo_matrix.col)))
    values = pt.FloatTensor(coo_matrix.data)
    shape = coo_matrix.shape
    sparse_tensor = pt.sparse_coo_tensor(indices, values, pt.Size(shape))
    return sparse_tensor.to(device=device)

def cosine(a, b=None):
    if b is None:
        b = a
    a_norms = np.einsum('ij,ij->i', a, a)
    b_norms = np.einsum('ij,ij->i', b, b)
    r = np.dot(a_norms, b_norms.T)
    return r

def compute_similarity(files=None, type='cosine', hems='L', sparse=True, mean_centering=True):
    dist = []
    if files is None:
        file_name = '../data/group.%s.wbeta.32k.func.gii' % hems
    else:
        file_name = files
    if type == 'cosine':
        mat = nb.load(file_name)
        data = [x.data for x in mat.darrays]
        data = np.reshape(data, (len(data), len(data[0])))
        data = data.transpose()
        if mean_centering:
            mean = data.mean(axis=1)
            data = data - mean[:, np.newaxis]
        else:
            data = data
        dist = cosine(data)
    elif type == 'pearson':
        mat = nb.load(file_name)
        data = [x.data for x in mat.darrays]
        data = np.reshape(data, (len(data), len(data[0])))
        data = data.transpose()
        if mean_centering:
            mean = data.mean(axis=1)
            data = data - mean[:, np.newaxis]
        else:
            data = data
        dist = np.corrcoef(data)
    return scipy.sparse.csr_matrix(dist) if sparse else dist

def compute_dist(coord, backend='torch'):
    if backend == 'torch' and TORCH_AVAILABLE:
        if type(coord) is np.ndarray:
            coord = pt.tensor(coord, dtype=pt.get_default_dtype())
        assert type(coord) is pt.Tensor, 'Input coord must be pytorch tensor!'
        return compute_dist_pt(coord)
    elif backend == 'numpy' or not TORCH_AVAILABLE:
        return compute_dist_np(coord)
    else:
        raise ValueError('Torch not available and no valid backend specified!')

def compute_dist_pt(coord):
    if type(coord) is np.ndarray:
        coord = pt.tensor(coord, dtype=pt.get_default_dtype())
    num_points = coord.shape[0]
    D = pt.zeros((num_points, num_points))
    for i in range(3):
        D = D + (coord[:, i].reshape(-1, 1) - coord[:, i]) ** 2
    return pt.sqrt(D)

def compute_dist_np(coord):
    num_points = coord.shape[0]
    D = np.zeros((num_points, num_points))
    for i in range(3):
        D = D + (coord[:, i].reshape(-1, 1) - coord[:, i]) ** 2
    return np.sqrt(D)

def compute_var_cov(data, cond='all', mean_centering=True, backend='torch'):
    if backend == 'torch' and TORCH_AVAILABLE:
        if type(data) is np.ndarray:
            data = pt.tensor(data, dtype=pt.get_default_dtype())
        assert type(data) is pt.Tensor, 'Input data must be pytorch tensor!'
        return compute_var_cov_pt(data, cond=cond, mean_centering=mean_centering)
    elif backend == 'numpy' or not TORCH_AVAILABLE:
        return compute_var_cov_np(data, cond=cond, mean_centering=mean_centering)
    else:
        raise ValueError('Torch not available and no valid backend specified!')

def compute_var_cov_np(data, cond='all', mean_centering=True):
    if mean_centering:
        mean = data.mean(axis=1)
        data = data - mean[:, np.newaxis]
    else:
        data = data
    if cond != 'all':
        data = data[:, cond]
    elif cond == 'all':
        data = data
    else:
        raise TypeError("Invalid condition type input! cond must be either 'all' or the column indices of expected task conditions")
    k = data.shape[1]
    sd = np.sqrt(np.sum(np.square(data), axis=1) / (k - 1))
    sd = np.reshape(sd, (sd.shape[0], 1))
    var = np.matmul(sd, sd.transpose())
    cov = np.matmul(data, data.transpose()) / (k - 1)
    return (cov, var)

def compute_var_cov_pt(data, cond='all', mean_centering=True):
    if mean_centering:
        data = data - pt.mean(data, dim=1, keepdim=True)
    if cond != 'all':
        data = data[:, cond]
    elif cond == 'all':
        data = data
    else:
        raise TypeError("Invalid condition type input! cond must be either 'all' or the column indices of expected task conditions")
    k = data.shape[1]
    cov = pt.matmul(data, data.T) / (k - 1)
    sd = pt.sqrt(pt.sum(data ** 2, dim=1, keepdim=True) / (k - 1))
    var = pt.matmul(sd, sd.T)
    return (cov, var)

def plot_single(within, between, subjects, maxDist=35, binWidth=1, within_color='k', between_color='r'):
    fig = plt.figure()
    numBins = int(np.floor(maxDist / binWidth))
    num_sub = len(subjects)
    x = np.arange(0, maxDist, binWidth) + binWidth / 2
    y_within = within.reshape(num_sub, -1)
    y_between = between.reshape(num_sub, -1)
    plt.errorbar(x, y_within.mean(0), yerr=y_within.std(0), ecolor=within_color, color=within_color, elinewidth=0.5, capsize=2, linestyle='dotted', label='within')
    plt.errorbar(x, y_between.mean(0), yerr=y_between.std(0), ecolor=between_color, color=between_color, elinewidth=0.5, capsize=2, linestyle='dotted', label='between')
    plt.legend(loc='upper right')
    plt.show()

def plot_wb_curve(T, path, sub_list=None, hems='all', within_color='k', between_color='r'):
    fig = plt.figure()
    bin_width = [value for (key, value) in T.items()][0]['binWidth']
    max_dist = [value for (key, value) in T.items()][0]['maxDist']
    k = len([value for (key, value) in T.items()][0]['corr_within'])
    x = np.arange(0, max_dist, bin_width) + bin_width / 2
    if sub_list is not None:
        subjects_dir = sub_list
    else:
        subjects_dir = scan_subdirs(path)
    (y_within, y_between) = (np.empty([1, k]), np.empty([1, k]))
    for sub in subjects_dir:
        data = [value for (key, value) in T.items() if sub in key]
        if len(data) == 2 and hems == 'all':
            within = (np.asarray(data[0]['corr_within']) + np.asarray(data[1]['corr_within'])) / 2
            between = (np.asarray(data[0]['corr_between']) + np.asarray(data[1]['corr_between'])) / 2
        elif len(data) == 1 and data[0]['hemisphere'] is hems:
            within = data[0]['corr_within']
            between = data[0]['corr_between']
        else:
            raise Exception('Incomplete DCBC evaluation. Missing result of %s.' % sub)
        y_within = np.vstack((y_within, within))
        y_between = np.vstack((y_between, between))
        sub_list = T.keys()
    y_within = np.delete(y_within, 0, axis=0)
    y_between = np.delete(y_between, 0, axis=0)
    plt.errorbar(x, y_within.mean(0), yerr=y_within.std(0), ecolor=within_color, color=within_color, label='within')
    plt.errorbar(x, y_between.mean(0), yerr=y_between.std(0), ecolor=between_color, color=between_color, label='between')
    plt.xlabel('spatial distance')
    plt.ylabel('Correlation between vertex pairs')
    plt.legend(loc='upper right')
    plt.show()
