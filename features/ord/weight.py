from ml.feature import CatFeatureDescriptor
from sklearn.preprocessing import OrdinalEncoder

# TODO 98k Nones among 100k

categories = ['[0-25)', '[25-50)', '[50-75)', '[75-100)', '[100-125)',
       '[125-150)', '[150-175)', '[175-200)', '>200']
fd = CatFeatureDescriptor(name='weight', categories=categories)
# fd.steps[1] = [(OrdinalEncoder, {'categories': [[categories]]})]  # TODO if change steps, unk_val is not appended
