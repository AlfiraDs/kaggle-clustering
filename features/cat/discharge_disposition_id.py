from ml.feature import CatFeatureDescriptor

categories = ['25', '1', '3', '6', '2', '5', '11', '7', '10', '4', '14', '18',
              '8', '13', '12', '16', '17', '22', '23', '9', '20', '15', '24',
              '28', '19', '27']
fd = CatFeatureDescriptor('discharge_disposition_id', categories=categories)
