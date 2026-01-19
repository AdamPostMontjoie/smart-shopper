
        return recipe_response.data
    except Exception as e:
        print(f"Failed to get recipes: {e}")
        return []
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text