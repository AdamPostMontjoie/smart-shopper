import os
import logging
from dotenv import load_dotenv
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from supabase import create_client, Client
from groq import Groq

load_dotenv()


url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(url, key)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_recipes_from_db(protein, calories, count:int):
    try:
        recipe_response = supabase. \
            rpc('recommend_recipes',
                {'min_protein_g':protein,
                'max_calories':calories,
                'min_match_percent':.50,
                'limit_count':count
                }). \
                execute()
        return recipe_response.data
    except Exception as e:
        print(f"Failed to get recipes: {e}")
        return []
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    
    # --- SIMPLE INTENT PARSING (You can make this smarter later) ---
    # Default values
    target_protein = 20
    target_cals = 1000
    
    # If user mentions numbers, try to grab them (Very basic logic)
    words = user_text.split()
    for w in words:
        if w.isdigit():
            val = int(w)
            if val < 100: target_protein = val # Assume small number is protein
            else: target_cals = val           # Assume big number is calories
    
    await update.message.reply_text(f" Searching for deals: >{target_protein}g protein, <{target_cals} cals...")

    # A. RETRIEVE (Get facts from DB)
    recipes = get_recipes_from_db(target_protein, target_cals,5)
    
    if not recipes:
        await update.message.reply_text("I couldn't find any recipes on sale matching those macros right now.")
        return

    # B. AUGMENT (Create the prompt)
    # We turn the database rows into a text block for the LLM
    context_str = ""
    for r in recipes:
        context_str += f"""
        - Recipe: {r['name']}
          Macros: {r['protein_g']}g Protein, {r['calories']} Calories
          On Sale: {r['ingredients_on_sale']} out of {r['total_ingredients']} ingredients are on sale!
          Instructions (Brief): {r['instructions'][:200]}...
        """

    system_prompt = f"""
    You are a thrifty shopping assistant. I have found the following recipes that are ON SALE at the grocery store right now.
    Your job is to recommend them to the user based on their request: "{user_text}".
    
    DATA (Use ONLY this):
    {context_str}
    
    Rules:
    1. Be enthusiastic about the savings.
    2. Explicitly mention the protein/calorie counts from the data.
    3. Do not make up recipes not in the list.
    """

    # C. GENERATE (Call LLM)
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ],
        model="llama3-70b-8192", # Fast and smart
    )

    ai_reply = chat_completion.choices[0].message.content
    await update.message.reply_text(ai_reply)

if __name__ == '__main__':
    app = ApplicationBuilder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()
    
    # Handlers
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), chat))
    
    print("Bot is polling...")
    app.run_polling()