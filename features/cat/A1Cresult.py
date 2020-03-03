from ml.feature import CatFeatureDescriptor

# TODO 96000 of Nones
categories = ['>7', '>8', 'Norm']
fd = CatFeatureDescriptor('A1Cresult', categories=categories)
