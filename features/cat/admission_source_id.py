from ml.feature import CatFeatureDescriptor

categories = ['1', '7', '2', '4', '5', '6', '20', '3', '17', '8', '9', '14',
              '10', '22', '11', '25', '13']
fd = CatFeatureDescriptor('admission_source_id', categories=categories)
