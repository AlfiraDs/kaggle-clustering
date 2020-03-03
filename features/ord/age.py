from ml.feature import CatFeatureDescriptor
from sklearn.preprocessing import OrdinalEncoder

categories = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)',
              '[90-100)']
fd = CatFeatureDescriptor(name='age', categories=categories)
fd.steps[1] = [(OrdinalEncoder, {'categories': [[categories]]})]
