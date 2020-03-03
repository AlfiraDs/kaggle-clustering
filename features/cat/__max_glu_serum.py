from ads.utils.feature import CatFeatureDescriptor

# TODO 96000 of Nones
categories = ['>300', 'Norm', '>200']
fd = CatFeatureDescriptor('max_glu_serum', categories=categories)
