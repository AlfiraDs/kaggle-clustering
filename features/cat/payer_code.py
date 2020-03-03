from ads.utils.feature import CatFeatureDescriptor

# TODO half of the data are Nones
categories = ['MC', 'MD', 'HM', 'UN', 'BC', 'SP', 'CP', 'SI', 'DM', 'CM',
       'CH', 'PO', 'WC', 'OT', 'OG', 'MP', 'FR']
fd = CatFeatureDescriptor('payer_code', categories=categories)
