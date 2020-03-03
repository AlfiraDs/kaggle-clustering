import os

from data import Data


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
FEATURES_DIR = os.path.join(os.path.dirname(__file__), 'features')

dtype = {col: str for col in [
    'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
]}

d = Data(data_dir=DATA_DIR, target_col='readmitted_NO', dtype=dtype)
data = d.data
train = d.train
y = d.y
test = d.test
