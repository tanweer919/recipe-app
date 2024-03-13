import pandas as pd
import os
import re
import isodate
import pprint
def convert_iso_duration(duration_str):
    duration = isodate.parse_duration(duration_str)
    
    days = duration.days
    hours = duration.seconds // 3600
    minutes = (duration.seconds % 3600) // 60
    
    components = []
    if days:
        components.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        components.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        components.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    
    return ", ".join(components)
cwd = os.getcwd()
df = pd.read_parquet(cwd + '/recipe_app/data/recipe_and_nutrition.parquet')
df = df.iloc[:1]
pprint.pprint(df.iloc[0].to_dict(), depth=None)