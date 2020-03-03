from ml.feature import CatFeatureDescriptor

categories = ['No', 'Up', 'Steady', 'Down']
fd = CatFeatureDescriptor('glyburide-metformin', categories=categories)
