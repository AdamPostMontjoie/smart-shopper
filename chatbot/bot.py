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

class IngredientDetail(TypedDict):
    ingredient_name: str
    is_on_sale: bool
    deal_name: str | None
    price: float | None
    discount: str | None
    liked_count:int

class Recipe(TypedDict):
    name:str
    all_ingredients:List[IngredientDetail]
    protein_g:float
    calories:float
    
    ingredient_on_sale:int
    total_ingredients:int

class ShopperState(TypedDict):
    user_id:str
    user_text:str #can change later to list once longer conversations
    user_intent:str
    recipes: List[Recipe]
    matched_recipes:Annotated[list,operator.add]
    dislikes:Annotated[list,operator.add]
    likes:Annotated[list,operator.add]
    supabase_offset:int #for scrolling if first 50 don't work out
    final_response:str



# structured llm calls
class UpdatePreferences(BaseModel):
    new_dislikes: List[str] = Field(description="List of ingredients the user dislikes/does not want")
    new_likes:List[str] = Field(description="List of ingredients the user likes/wants")
class UserIntent(BaseModel):
    intent: Literal["recipe", "profile", "other"] = Field(
        description="Classify the user's main goal. 'recipe' if they want food suggestions. 'profile' if they are ONLY updating preferences by stating dislikes/food they can't eat. 'other' for greeting/help."
)
class FilterResult(BaseModel):
    safe_indices: List[int] = Field(
    description="The indices (0-based) of the recipes that are SAFE to eat (do NOT contain disliked ingredients)."
)
class RecipeLikeScore(BaseModel):
    recipe_index: int = Field(description="The index of the recipe from the provided list.")
    liked_count: int = Field(description="The number of user likes that are present in this recipe's ingredients.")
class LikeScorerResult(BaseModel):
    scores: List[RecipeLikeScore] = Field(description="List of scores for each recipe.")
    
class PrettyResponse(BaseModel):
    recipe_text: str = Field(
        description="A pretty list of recipes and ingredients to shop for"
    )



#extract any dislikes for filtering
def profile_node(state:ShopperState):
    user_text = state['user_text']
    user_id = state['user_id']
    system_prompt = """
    You are an expert at extracting dietary restrictions.
    
    Analyze the user's input. Extract ONLY ingredients the user explicitly:
    - Dislikes
    - Is allergic to
    - Wants to avoid
    
    CRITICAL RULES:
    1. If the user asks FOR an ingredient (e.g., "I want chicken"), DO NOT include it here.
    2. If the user specifies NO negative preferences, return an empty list [].
    3. Do not infer dislikes. Only extract what is explicitly stated.
    """
    response = llm.with_structured_output(UpdatePreferences).invoke([
        ("system", system_prompt),
        ("human",user_text)
    ])
    if response.new_dislikes or response.new_likes:
        updated_dislikes = list(set(state['dislikes'] + response.new_dislikes))
        updated_likes = list(set(state['likes'] + response.new_likes))
        if user_id:
            try:
                supabase.table("user_preferences").upsert({
                    "user_id": user_id, 
                    "dislikes": updated_dislikes,
                    "likes": updated_likes
                }).execute()
                print(f"Saved prefs for {user_id}")
            except Exception as e:
                print(f"Save Error: {e}")
    return {"dislikes":response.new_dislikes, "likes":response.new_likes}

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
            rpc('recommend_recipes1',
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
    likes = state.get('likes',[])
    
    # skip llm call if no dislikes
    if dislikes and recipes:
        #create numbered list
        dislike_batch_text = ""
        for i, r in enumerate(recipes):
            # Extract ingredient names from the sale details
            ing_list = [deal['deal_name'] for deal in r.get('all_ingredients', [])]
            dislike_batch_text += f"ID {i}: {r['name']} | Contains: {', '.join(ing_list)}\n"
            
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
                ("human", dislike_batch_text)
            ])
            
            valid_recipes = [recipes[i] for i in response.safe_indices if i < len(recipes)]
            print(f"Filter removed {len(recipes) - len(valid_recipes)} recipes.")

        except Exception as e:
            print(f"Filter Error: {e}. Returning original list.")
            return {"recipes": recipes}
    else:
        valid_recipes = recipes
    if likes and recipes:
        like_batch_text = ""
        for i, r in enumerate(valid_recipes):
            ing_list = [ing.get('ingredient_name', '') for ing in r.get('all_ingredients', [])]
            like_batch_text += f"ID {i}: {r['name']} | Contains: {', '.join(ing_list)}\n"
        print(f"Scoring {len(valid_recipes)} safe recipes against likes: {likes}")
        likes_system_prompt = f"""
        You are a semantic matching assistant. 
        The user explicitly LIKES these ingredients: {', '.join(likes)}
        
        For each recipe provided, count how many of the user's "likes" are present in the recipe's ingredients. 
        Match semantically (e.g., if user likes "chicken", then "chicken breast" counts as a match).
        Do not count the same "like" multiple times for a single recipe.
        """
        try:
            likes_response = llm.with_structured_output(LikeScorerResult).invoke([
                ("system", likes_system_prompt),
                ("human", like_batch_text)
            ])
            
            # Map the LLM's scores back to our recipe dictionaries
            score_map = {score.recipe_index: score.liked_count for score in likes_response.scores}
            
            for i, r in enumerate(valid_recipes):
                r['liked_count'] = score_map.get(i, 0)
                
            # Sort the recipes by the LLM's liked_count in descending order (highest at the top)
            valid_recipes.sort(key=lambda x: x.get('liked_count', 0), reverse=True)
            print("Successfully sorted recipes by LLM like-scores.")

        except Exception as e:
            print(f"Likes Scoring Error: {e}. Proceeding without sorting by likes.")
    return {"matched_recipes": valid_recipes}
#"if less than 3 recipes remain after filtering, go back to db node and get next 50"
def filtered_conditional(state:ShopperState):
    recipes = state['matched_recipes']
    offset = state['supabase_offset']

    print(f"Searching for recipes with offset of {offset}")
    if len(recipes) <=3 and offset <= 250:
        return "database_node"
    else:
        return "final_recipes_node"
    
#return final list in chat format
def final_recipes_node(state:ShopperState):
    # Limit to 5 responses
    matched_recipes = state.get('matched_recipes')[:5]
    if len(matched_recipes) > 0:
        #Distinguish clearly between 'On Sale' items and 'Regular Price' items. (include when available)
        system_prompt = "You are a helpful shopping assistant. Present these meal options nicely. Ensure the recipe names remove any irregularities(names, unneeded numbers) while still mantaining descriptiveness. Group the shopping list by category if possible. "
        response = llm.with_structured_output(PrettyResponse).invoke([
            ("system",system_prompt),
            ("human",str(matched_recipes))
        ])
            
        return {"final_response": response.recipe_text}
    else:
        return {"final_response":"Sorry, I couldn't find you any recipes from this weeks deals based on your profile"}



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
                "dislikes": [],
                "likes":[]
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
        raw_likes = response.data[0]['likes'] if response.data else []
        existing_dislikes = list(set([d for d in raw_dislikes if d and d.strip()])) #strips whitespace 
        existing_likes = list(set([d for d in raw_likes if d and d.strip()]))
    except Exception as e:
        print(f"Memory Fetch Error: {e}")
        existing_dislikes = []
    
    # 1. Initialize State
    initial_state = {
        "user_id":user_id,
        "user_text": user_text,
        "user_intent": "",
        "recipes": [],
        "matched_recipes": [],
        "dislikes": existing_dislikes, 
        "likes":existing_likes,
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