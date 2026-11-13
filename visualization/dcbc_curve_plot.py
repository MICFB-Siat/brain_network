import numpy as np

def plot_wb_curve(T, path, sub_list=None, hems='all', within_color='k', between_color='r', ax=None, sem=True, label='', lw=1.0, capsize=0, zorder=2):
    bin_width = [value for (key, value) in T.items()][0]['binWidth']
    max_dist = [value for (key, value) in T.items()][0]['maxDist']
    k = len([value for (key, value) in T.items()][0]['corr_within'])
    x = np.arange(0, max_dist, bin_width) + bin_width / 2
    if sub_list is not None:
        subjects_dir = sub_list
    else:
        from DCBC.utilities import scan_subdirs
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
    n = y_within.shape[0]
    err_w = y_within.std(0, ddof=1) / np.sqrt(n) if sem else y_within.std(0)
    err_b = y_between.std(0, ddof=1) / np.sqrt(n) if sem else y_between.std(0)
    ax.errorbar(x, y_within.mean(0), yerr=err_w, ecolor=within_color, color=within_color, label=f'{label} within', linestyle='-', lw=lw, elinewidth=lw * 0.8, capsize=capsize, zorder=zorder)
    ax.errorbar(x, y_between.mean(0), yerr=err_b, ecolor=between_color, color=between_color, label=f'{label} between', linestyle='--', lw=lw, elinewidth=lw * 0.8, capsize=capsize, zorder=zorder)
    return (x, y_within.mean(0), y_between.mean(0))
