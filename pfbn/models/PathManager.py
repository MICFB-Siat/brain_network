class PathManager:

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('PathManager stub: original is unreadable; always pass dataset_path explicitly')

    def get_tmp_path(self, dataset_name, create=False):
        raise NotImplementedError('PathManager stub')
