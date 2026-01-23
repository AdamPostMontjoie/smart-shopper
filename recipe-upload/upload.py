import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from supabase import create_client, Client
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import ast
import time

load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

if not url or not key:
    raise ValueError("Missing Supabase credentials. Check .env")

supabase: Client = create_client(url, key)

model = SentenceTransformer('all-MiniLM-L6-v2')

# --- CONFIGURATION: ANCHOR MAPPING ---
# Instead of generic names, we use specific examples to "ground" the vectors.
CATEGORY_ANCHORS = {
    'Bakery': ['brioche','buns',  'bagel', 'roll', 'cake', 'pie', 'cookie', 'muffin', 'donut', 'croissant', 'tortilla'],
    'Beer & Wine': ['beer', 'wine', 'alcohol', 'liquor', 'ipa', 'ale', 'chardonnay'],
    'Cheese Shoppe': ['cheese', 'brie', 'gouda', 'aged cheddar', 'parmesan', 'blue cheese', 'goat cheese', 'specialty cheese'],
    'Dairy & Frozen Foods': ['milk', 'butter', 'yogurt', 'cream', 'egg', 'ice cream', 'frozen pizza', 'frozen dinner', 'sorbet'],
    'Delicatessen': ['deli meat', 'sliced turkey', 'sliced ham', 'prosciutto', 'salami', 'roast beef', 'potato salad'],
    'Grocery': ['bread', 'cereal', 'pasta', 'rice', 'peppercorn','salt' 'beans', 'sauce', 'oil', 'spice', 'chips', 'candy', 'soda', 'canned soup', 'flour', 'sugar','vinegar', 'peanut butter', 'nuts', 'crackers', 'juice', 'water'],
    "Market's Café": ['prepared sandwich', 'soup', 'salad bar', 'hot coffee', 'slice of pizza'],
    "Market's Kitchen": ['rotisserie chicken', 'fried chicken', 'hot tenders', 'cooked sides', 'macaroni and cheese'],
    'Meat': ['beef', 'pork', 'chicken breast', 'steak', 'ground beef', 'sausage', 'bacon', 'lamb', 'ribs'],
    'Produce': ['fruit', 'vegetable', 'apple', 'banana', 'lettuce', 'tomato', 'potato', 'onion', 'pepper', 'mushroom', 'berry', 'citrus', 'herb', 'garlic'],
    'Seafood': ['fish', 'shrimp', 'salmon', 'tuna', 'crab', 'lobster', 'scallop', 'cod', 'tilapia', 'shellfish'],
    'Sushi': ['sushi', 'sashimi', 'spicy tuna roll', 'california roll', 'seaweed', 'wasabi']
}

# Flatten the anchors into two lists for vectorization
# anchor_terms: ["bread", "bagel", ..., "beer", "wine", ...]
# anchor_map:   ["Bakery", "Bakery", ..., "Beer & Wine", "Beer & Wine", ...]
anchor_terms = []
anchor_map = []

for category, keywords in CATEGORY_ANCHORS.items():
    for k in keywords:
        anchor_terms.append(k)
        anchor_map.append(category)

print(">>> Vectorizing Anchors...")
# We encode the ~100 specific terms instead of the 12 category names
anchor_embeddings = model.encode(anchor_terms)


def safe_execute(query_obj, retries=3):
    for i in range(retries):
        try:
            return query_obj.execute()
        except Exception as e:
            if i == retries - 1: raise e
            time.sleep(2 * (i + 1)) 

def parse_col(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return None
    return val

def get_nutrient(nutrients_dict, key):
    if not isinstance(nutrients_dict, dict): return 0
    item = nutrients_dict.get(key)
    if isinstance(item, dict):
        return item.get('quantity', 0)
    return 0

# --- MAIN SCRIPT ---
print(">>> 1. Loading Data...")
ds = load_dataset("datahiveai/recipes-with-nutrition")
df = ds['train'].to_pandas()

print(">>> 2. Parsing Columns...")
df['ingredients'] = df['ingredients'].apply(parse_col)
df['total_nutrients'] = df['total_nutrients'].apply(parse_col)
df['ingredient_lines'] = df['ingredient_lines'].apply(parse_col)

print(">>> 3. Filtering...")
keep_mask = []
for index, row in df.iterrows():
    protein = get_nutrient(row['total_nutrients'], 'PROCNT')
    if protein > 15 and isinstance(row['ingredients'], list) and len(row['ingredients']) > 0:
        keep_mask.append(True)
    else:
        keep_mask.append(False)

df = df[keep_mask]
print(f"    Recipes remaining: {len(df)}")

print(">>> 4. Vectorizing Ingredients...")
unique_ing_set = set()
for ing_list in df['ingredients']:
    for item in ing_list:
        if isinstance(item, dict) and 'food' in item:
            unique_ing_set.add(item['food'].lower().strip())

unique_ing_list = list(unique_ing_set)
ing_embeddings = model.encode(unique_ing_list)

# --- CALCULATE CATEGORY ---
print(">>> Calculating Categories...")
# Measure distance from Ingredient -> Every Anchor Term
# Shape: (Num_Ingredients, Num_Anchors)
dists = cdist(ing_embeddings, anchor_embeddings, metric='cosine')

# --- UPLOAD PHASE 1: INGREDIENTS ---
print(f">>> 5. Uploading {len(unique_ing_list)} Ingredients...")
ing_payload = []

for i, (name, emb) in enumerate(zip(unique_ing_list, ing_embeddings)):
    # 1. Find the index of the closest specific anchor (e.g., "peanut butter")
    closest_anchor_idx = np.argmin(dists[i])
    
    # 2. Map that anchor back to its category (e.g., "Grocery")
    assigned_category = anchor_map[closest_anchor_idx]
    
    ing_payload.append({
        "name": name, 
        "embedding": emb.tolist(), 
        "category": assigned_category
    })

name_to_id_map = {}
batch_size = 200 

for i in range(0, len(ing_payload), batch_size):
    batch = ing_payload[i:i+batch_size]
    try:
        query = supabase.table('unique_ingredients').upsert(batch, on_conflict="name")
        res = safe_execute(query)
        for item in res.data:
            name_to_id_map[item['name']] = item['id']
        print(f"    Uploaded batch {i} - {i+len(batch)}")
    except Exception as e:
        print(f"    CRITICAL FAIL on batch {i}: {e}")
        continue 

# --- UPLOAD PHASE 2: RECIPES (BATCHED) ---
print(">>> 6. Uploading Recipes (Batch Mode)...")

recipe_batch_size = 50 
total_rows = len(df)

for i in range(0, total_rows, recipe_batch_size):
    chunk = df.iloc[i : i+recipe_batch_size]
    recipes_to_insert = []
    
    for _, row in chunk.iterrows():
        nutrients = row['total_nutrients']
        recipes_to_insert.append({
            "name": row['recipe_name'],
            "protein_g": get_nutrient(nutrients, 'PROCNT'),
            "calories": get_nutrient(nutrients, 'ENERC_KCAL'),
            "instructions": str(row['ingredient_lines']), 
            "display_ingredients": row['ingredient_lines'], 
            "image_url": row['image_url']
        })
    
    try:
        res = safe_execute(supabase.table('recipes').insert(recipes_to_insert))
        inserted_recipes = res.data
        junctions_to_insert = []
        
        for db_row, (_, original_row) in zip(inserted_recipes, chunk.iterrows()):
            recipe_id = db_row['id']
            seen_ingredients = set()
            
            for item in original_row['ingredients']:
                if not isinstance(item, dict) or 'food' not in item: continue
                ing_name = item['food'].lower().strip()
                if ing_name in name_to_id_map:
                    ing_id = name_to_id_map[ing_name]
                    if ing_id not in seen_ingredients:
                        junctions_to_insert.append({
                            "recipe_id": recipe_id,
                            "ingredient_id": ing_id
                        })
                        seen_ingredients.add(ing_id)
        
        if junctions_to_insert:
            safe_execute(supabase.table('recipe_ingredients').upsert(junctions_to_insert, ignore_duplicates=True))
        print(f"    ✅ Uploaded Recipes {i} - {i+len(chunk)}")
        
    except Exception as e:
        print(f"    CRITICAL ERROR on Recipe Batch {i}: {e}")
        continue

print("\n>>> UPLOAD COMPLETE.")