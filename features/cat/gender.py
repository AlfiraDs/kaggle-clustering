from ml.feature import CatFeatureDescriptor

categories = ['Male', 'Female', 'Unknown/Invalid']
fd = CatFeatureDescriptor('gender', categories=categories)
