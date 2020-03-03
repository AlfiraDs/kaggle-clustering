from ml.feature import CatFeatureDescriptor

categories = ['Caucasian', 'AfricanAmerican', 'Other', 'Asian', 'Hispanic']
fd = CatFeatureDescriptor('race', categories=categories)
