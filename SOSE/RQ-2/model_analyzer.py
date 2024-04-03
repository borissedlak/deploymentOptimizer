import ast

import pandas as pd

precision_df = pd.read_csv('./precision_inference.csv')
precision_df['matrix'] = precision_df['matrix'].apply(ast.literal_eval)

print(precision_df)
