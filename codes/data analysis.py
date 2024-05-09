import pandas as pd
credit_card = pd.read_csv('../creditcard data/creditcard.csv')
print(credit_card.head(),'\n',
      credit_card.head(20),'\n',
      credit_card.shape,'\n',
      credit_card.info(),'\n',
      credit_card.isnull().sum(),'\n',
      credit_card['Class'].nunique(),'\n',
      credit_card.Class.value_counts())