import pandas as pd
import supabase 
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

ds = load_dataset("datahiveai/recipes-with-nutrition")

recipe_df = ds['train'].to_pandas()
print(recipe_df[[ 'ingredients']].head(10))
#low protein removed
recipe_df = recipe_df[recipe_df['total_nutrients'].apply(lambda x: x['PROCNT']['quantity'] > 15)]
