from ml.feature import CatFeatureDescriptor

categories = ['No', 'Steady', 'Up', 'Down']
fd = CatFeatureDescriptor('metformin', categories=categories)
