import pandas as pd
import os
import re
cwd = os.getcwd()
df = pd.read_parquet(cwd + '/recipe_app/data/recipes_w_search_terms.parquet')
df1 = pd.read_parquet(cwd + '/recipe_app/data/recipes.parquet')
df1 = df1.rename(columns={'RecipeId': 'id'})
result = df.merge(df1[['id','CookTime', 'PrepTime', 'TotalTime', 'Images', 'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent']], how='left', on='id')
result = result.sort_values(by=['id'])
result = result[~result['name'].str.contains('pork', na=False, flags=re.IGNORECASE)]
result = result[~result['name'].str.contains('ham', na=False, flags=re.IGNORECASE)]
result.to_parquet(cwd + '/recipe_app/data/recipe_and_nutrition.parquet')