from ml.feature import CatFeatureDescriptor

categories = ['No', 'Up', 'Steady', 'Down']
fd = CatFeatureDescriptor('chlorpropamide', categories=categories)
