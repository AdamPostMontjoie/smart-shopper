import os
import logging
from dotenv import load_dotenv
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from supabase import create_client, Client
from groq import Groq
import operator
from pydantic import BaseModel, Field
from typing import Annotated, List, Literal
from typing_extensions import TypedDict

from langchain_community.document_loaders import WikipediaLoader
from langchain_groq import ChatGroq

from langgraph.graph import END, MessagesState, START, StateGraph

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(url, key)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
llm = ChatGroq(model="llama-3.3-70b-versatile")

class Sale(TypedDict):
    deal_name:str
    price:float

class Recipe(TypedDict):
    name:str
    sale_details:List[Sale]
    protein_g:float
    calories:float
    ingredient_on_sale:int
    total_ingredients:int

class ShopperState(TypedDict):
    user_id:str
    user_text:str #can change later to list once longer conversations
    user_intent:str
    recipes: List[Recipe]
    dislikes:Annotated[list,operator.add]
    supabase_offset:int #for scrolling if first 50 don't work out
    final_response:str



# structured llm calls
class UpdatePreferences(BaseModel):
    new_dislikes: List[str] = Field(description="List of ingredients the user dislikes")
class UserIntent(BaseModel):
    intent: Literal["recipe", "profile", "other"] = Field(
        description="Classify the user's main goal. 'recipe' if they want food suggestions. 'profile' if they are ONLY updating preferences by stating dislikes/food they can't eat. 'other' for greeting/help."
)
class FilterResult(BaseModel):
    safe_indices: List[int] = Field(
    description="The indices (0-based) of the recipes that are SAFE to eat (do NOT contain disliked ingredients)."
)
    
class PrettyResponse(BaseModel):
    recipeText: str = Field(
        description="A pretty list of recipes and ingredients to shop for"
    )



#extract any dislikes for filtering
def profile_node(state:ShopperState):
    user_text = state['user_text']
    user_id = state['user_id']
    response = llm.with_structured_output(UpdatePreferences).invoke([
        ("system", "Extract any ingredients the user dislikes or cannot eat."),
        ("human",user_text)
    ])
    if response.new_dislikes:
        updated_list = list(set(state['dislikes'] + response.new_dislikes))
        if user_id:
            try:
                supabase.table("user_preferences").upsert({
                    "user_id": user_id, 
                    "dislikes": updated_list 
                }).execute()
                print(f"Saved prefs for {user_id}")
            except Exception as e:
                print(f"Save Error: {e}")
    return {"dislikes":response.new_dislikes}

def user_intent_node(state:ShopperState):
    user_text = state['user_text']
    #determine intent
    response = llm.with_structured_output(UserIntent).invoke([
        ("system", "You are a router. Determine if the user wants to search for recipes, just update their profile, or is chatting."),
        ("human",user_text)
    ])
    return {"user_intent":response.intent}

def intent_conditional(state:ShopperState):
    intent = state['user_intent']
    if intent == "recipe":
        return "database_node"
    return END



def database_node(state:ShopperState):
    protein = 40
    calories = 5000
    offset = state.get("supabase_offset", 0)
    print("attempting to grab recipes")
    try:
        recipe_response = supabase. \
            rpc('recommend_recipes',
                {'min_protein_g':protein,
                'max_calories':calories,
                'min_match_percent':.20,
                'limit_count': 50,
                'offset_val':offset
                }). \
                execute()
        return {
            "recipes":recipe_response.data or [],
            "supabase_offset": offset + 50
            }
    except Exception as e:
        print(f"Failed to get recipes: {e}")
        return {"recipes":[]}
#here i filter to make sure the recipes don't contain disliked ingredients
#greater filtering will be added if too many recipes pass first filter test
def filter_node(state: ShopperState):
    recipes = state.get('recipes', [])
    dislikes = state.get('dislikes', [])
    
    # skip llm call if no dislikes
    if not dislikes or not recipes:
        return {"recipes": recipes}

    #create numbered list
    batch_text = ""
    for i, r in enumerate(recipes):
        # Extract ingredient names from the sale details
        ing_list = [deal['deal_name'] for deal in r.get('sale_details', [])]
        batch_text += f"ID {i}: {r['name']} | Contains: {', '.join(ing_list)}\n"
        
    print(f"Filtering {len(recipes)} recipes against: {dislikes}")
    system_prompt = f"""
    You are a strict dietary safety filter.
    User Dislikes: {', '.join(dislikes)}
    
    Analyze the provided recipes. Return the INDICES of the recipes that are SAFE.
    
    Rules:
    1. If a recipe contains a disliked ingredient (even a derivative, e.g., 'cream' when user hates 'milk'), OMIT the index.
    2. If unsure, err on the side of caution and OMIT the index.
    3. Return ONLY the list of integers.
    """
    
    try:
        response = llm.with_structured_output(FilterResult).invoke([
            ("system", system_prompt),
            ("human", batch_text)
        ])
        
        valid_recipes = [recipes[i] for i in response.safe_indices if i < len(recipes)]
        print(f"Filter removed {len(recipes) - len(valid_recipes)} recipes.")
        return {"recipes": valid_recipes}

    except Exception as e:
        print(f"Filter Error: {e}. Returning original list.")
        return {"recipes": recipes}

#"if less than 3 recipes remain after filtering, go back to db node and get next 50"
def filtered_conditional(state:ShopperState):
    recipes = state['recipes']
    if len(recipes) == 0:
        return "final_recipes_node"
    if len(recipes) <3:
        return "database_node"
    else:
        return "final_recipes_node"
    
#return final list in chat format
def final_recipes_node(state:ShopperState):
    recipes = state['recipes'][:5] # Limit to 5 responses
    #Distinguish clearly between 'On Sale' items and 'Regular Price' items. (include when available)
    system_prompt = "You are a helpful shopping assistant. Present these meal options nicely. Group the shopping list by category if possible. "
    response = llm.with_structured_output(PrettyResponse).invoke([
        ("system",system_prompt),
        ("human",recipes)
    ])
        
    return {"final_response": response}



shopping_builder = StateGraph(ShopperState)
#nodes
shopping_builder.add_node(user_intent_node)
shopping_builder.add_node(profile_node)
shopping_builder.add_node(database_node)
shopping_builder.add_node(filter_node)
shopping_builder.add_node(final_recipes_node)


#flow
shopping_builder.add_edge(START, "profile_node")
shopping_builder.add_edge("profile_node", "user_intent_node")
shopping_builder.add_conditional_edges("user_intent_node",intent_conditional)
shopping_builder.add_edge("database_node", "filter_node")
shopping_builder.add_conditional_edges("filter_node",filtered_conditional)
shopping_builder.add_edge("final_recipes_node",END)

app_graph = shopping_builder.compile()

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    first_name = update.message.from_user.first_name
    
    print(f"Start command received from {user_id}")
    try:
        # Check if user already exists in DB
        response = supabase.table("user_preferences").select("user_id").eq("user_id", user_id).execute()
        
        # If no data returned, they are NEW
        if not response.data:
            welcome_message = (
                f"👋 Hi {first_name}! I'm your Smart Shopper Agent.\n\n"
                "I help you find high-protein recipes using ingredients currently on sale.\n\n"
                "**How to use me:**\n"
                "1. Tell me what you love: 'I love chicken!' \n"
                "2. Tell me what you hate: 'I hate mushrooms and cilantro'\n"
                "3. Ask for food: 'Give me chicken recipes'\n\n"
                "Let's get started! What are you looking for?"
            )
            
            # SAVE them to DB
            supabase.table("user_preferences").insert({
                "user_id": user_id, 
                "dislikes": []
            }).execute()
            
            await update.message.reply_text(welcome_message)
            
        else:
            await update.message.reply_text(f"Welcome back, {first_name}! Ready to cook?")
            
    except Exception as e:
        print(f"Start Handler Error: {e}")
        # Fallback in case DB fails
        await update.message.reply_text("👋 Hello! Ask me for recipes.")

async def telegram_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    #chat_id = update.message.chat_id
    user_id = update.message.from_user.id
    print(f"Received: {user_text}")
    try:
        response = supabase.table("user_preferences").select("dislikes").eq("user_id", user_id).execute()
        raw_dislikes = response.data[0]['dislikes'] if response.data else []
        existing_dislikes = list(set([d for d in raw_dislikes if d and d.strip()]))
    except Exception as e:
        print(f"Memory Fetch Error: {e}")
        existing_dislikes = []
    
    # 1. Initialize State
    initial_state = {
        "user_id":user_id,
        "user_text": user_text,
        "user_intent": "",
        "recipes": [],
        "dislikes": existing_dislikes, 
        "supabase_offset": 0,
        "final_response": ""
    }
    
    # 2. Run the Graph (Invoke)
    # This runs the whole flow we just built

    final_state = await app_graph.ainvoke(initial_state)
    
    # 3. Send the Result
    response = final_state.get("final_response")
    
    # Fallback if the graph ended early (e.g., just profile update)
    if not response:
        if final_state['user_intent'] == 'profile':
            response = "Got it! I've updated your preferences."
        else:
            response = "I'm mostly a shopping bot. Ask me for recipes!"
            
    await update.message.reply_text(response)

# --- RUN ---
if __name__ == '__main__':
    app = ApplicationBuilder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), telegram_handler))
    print("Bot is polling...")
    app.run_polling()