from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import time
import os
import re
import numpy as np
import json
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(url, key)

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_dynamic_schedule():
    options = webdriver.ChromeOptions()
    
    options.add_argument('--headless=new') # Use the new headless mode (more stable)
    options.add_argument('--no-sandbox') # Bypass OS security model (required for Docker/CI)
    options.add_argument('--disable-dev-shm-usage') # Overcome limited resource problems
    options.add_argument('--disable-gpu') # Applicable to windows os only but good practice
    options.add_argument('--window-size=1920,1080') # Prevent elements from being hidden
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        print("1. Opening website...")
        driver.get("https://www.shopmarketbasket.com/weekly-flyer/")
        
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "flyer_main"))
        )
        select_temp = Select(driver.find_element(By.ID, "ddlDepartments"))
        dept_options = [opt.text for opt in select_temp.options 
                    if "Loading" not in opt.text and "Featured" not in opt.text]
        print(f"Found {len(dept_options)} departments to scrape: {dept_options}")

        master_inventory = []   
        for dept_name in dept_options:
            print(f"\n>>> Switching to: {dept_name}")
        
            # Re-find the select element 
            select_element = Select(driver.find_element(By.ID, "ddlDepartments"))
            select_element.select_by_visible_text(dept_name)
            
    
            time.sleep(3) 
            click_counter = 0;
            while True and click_counter < 20:
                try:
                    #click load more button
                    click_counter += 1
                    load_more_btn = WebDriverWait(driver, 2).until(
                        EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, "Load More"))
                    )
                    driver.execute_script("arguments[0].scrollIntoView();", load_more_btn)
                    load_more_btn.click()
                    print("   [+] Clicked Load More")
                    time.sleep(1.5)
                except:
                    break
            

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            items = soup.find_all('li', class_='item')
            

            for item in items:
                try:
                    heading_div = item.find('div', class_='heading')
                    if not heading_div: continue
                    
                    name = heading_div.find('h2').get_text(strip=True)
                    
                    price_div = item.find('div', class_='price-holder')
                    price = price_div.find('h2').get_text(strip=True) if price_div else "N/A"
                    if "$" in price:
                        price = price.replace("$", "")
                    
                    if "¢" in price:
                        price = price.replace("¢", "")
                        price = f"0.{price}"
                    if "for" in price:
                        price = price.replace("for","")
                        parts = price.split()
                        if len(parts) >= 2 and parts[0].isdigit():
                            price = str(float(parts[1]) / float(parts[0]))
                        else:
                            raise ValueError(f"Bad 'for' price format: {price}")

                    

                    price = re.sub(r"[^0-9.]", "", price)
                    discount_div = item.find('div', class_="circle-deal")
                    discount = discount_div.find('p',class_='ng-binding').get_text(strip=True)
                    
                    master_inventory.append({
                        "name": name,
                        "price": price,
                        "discount":discount,
                        "category": dept_name
                    })
                except AttributeError:
                    continue

        print(f"\nScraping Complete. Total items: {len(master_inventory)}")
        print(list(filter(lambda x:x['category'] == "Meat", master_inventory)))

    except Exception as e:
        print(f"\nCRASHED: {e}")
        print("HTML at crash time:")
        print(driver.page_source[:500])
        
    finally:
        driver.quit()
        upload_new_deals(master_inventory)
def upload_new_deals(inventory):
    #match to unique ingredients
    ing_response = supabase.table('unique_ingredients').select('id','name','embedding','category').execute()
    ingredients = ing_response.data

    categorized_ingredients = {}
    for item in ingredients:
        if isinstance(item['embedding'],str):
            item['embedding'] = json.loads(item['embedding'])
        cat = item.get('category')
        
        if cat not in categorized_ingredients:
            categorized_ingredients[cat] = []
        
        # Add the ingredient to the correct list
        categorized_ingredients[cat].append(item)
    db_ids = [item['id'] for item in ingredients]
    
    
    deal_names = [item['name'] for item in inventory]
    deal_embeddings = model.encode(deal_names)
    cleaned_inventory = []

    for i, deal_item in enumerate(inventory):
        current_deal_vector = deal_embeddings[i]
        
        deal_cat = deal_item['category']
        
        best_match_id = None

        if deal_cat in categorized_ingredients:
            
            candidates = categorized_ingredients[deal_cat]
            
            candidate_vectors = [c['embedding'] for c in candidates]
            candidate_ids = [c['id'] for c in candidates]
            
            
            distances = cdist([current_deal_vector], candidate_vectors, metric='cosine')[0]
            
            closest_index = np.argmin(distances)       # Index of the lowest distance
            score = 1 - distances[closest_index]       # Convert distance (0 to 1) to Similarity (0% to 100%)
            
    
            if score > 0.60:
                best_match_id = candidate_ids[closest_index]
            

        # Save the results back to the item
        deal_item['ingredient_id'] = best_match_id
        deal_item['embedding'] = current_deal_vector.tolist() # Need to save as list for JSON

        # Cleaning step (Removing N/A, empty strings)
        clean_item = {}
        for key, value in deal_item.items():
            if isinstance(value, str) and value.strip() in ["", "N/A", "No Deal"]:
                clean_item[key] = None
            else:
                clean_item[key] = value
        
        cleaned_inventory.append(clean_item)
    try:
        supabase.table('deals').delete().neq("id",-1).execute()
        #upload deals
        try:
            supabase.table('deals').insert(cleaned_inventory).execute()
        except Exception as e:
            print(f"\nFailed to upload new deals: {e}")
    except Exception as e:
        print(f"\nFailed to delete old deals: {e}")
    
if __name__ == "__main__":
    get_dynamic_schedule()