from ml.feature import CatFeatureDescriptor

# TODO majority of one class
categories = ['No', 'Up', 'Steady', 'Down']
fd = CatFeatureDescriptor('metformin-pioglitazone', categories=categories)
