from ml.feature import CatFeatureDescriptor

# TODO the only 'No' value
categories = ['No', 'Up', 'Steady', 'Down']
fd = CatFeatureDescriptor('citoglipton', categories=categories)
