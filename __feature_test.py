import os

from utils import fit_transform_feature

import consts

if __name__ == '__main__':
    data = consts.data
    name = 'weight.py'.split('.')[0]
    fe_dir = os.path.join(consts.FEATURES_DIR)
    x = fit_transform_feature(name, data, fe_dir)
