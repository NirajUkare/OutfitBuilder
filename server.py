# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from openai import OpenAI
# import os
# from dotenv import load_dotenv
# import json

# load_dotenv()
# gemini_api_key = os.getenv("GEMINI_API_KEY")

# client = OpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/"
# )

# app = FastAPI(title="Outfit Builder API")

# class WishlistItem(BaseModel):
#     name: str
#     description: str
#     productId: str

# class Wishlist(BaseModel):
#     items: list[WishlistItem]

# def ask_gemini(wishlist):
#     prompt = f"""
# You are a fashion stylist AI. I will provide a wishlist containing items in JSON format.
# Your task is to build complete outfits using these items.  
# Each outfit should be an object containing multiple items that match well together.  
# Output must be in pure JSON only (no explanations).  

# Wishlist:
# {json.dumps([item.dict() for item in wishlist], indent=2)}

# Now return outfits in JSON format, where each outfit is an object with a list of matching items.
# """
#     response = client.chat.completions.create(
#         model="gemini-2.5-flash",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return response

# @app.post("/build-outfits")
# def build_outfits(wishlist: Wishlist):
#     try:
#         response = ask_gemini(wishlist.items)
#         outfits_raw = response.choices[0].message.content
#         outfits = json.loads(outfits_raw)  # parse Gemini's JSON into Python list/dict
#         return {"outfits": outfits}
#     except json.JSONDecodeError:
#         raise HTTPException(status_code=500, detail="Gemini did not return valid JSON")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))





from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from openai import OpenAI
import os
from dotenv import load_dotenv
import json

# --- Configuration ---
load_dotenv()
# It's good practice to validate that environment variables are set
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# Make the model name configurable
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash") # Changed to gemini-pro for better JSON generation, but you can use any compatible model

client = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

app = FastAPI(
    title="Outfit Builder API",
    description="An API that uses a Gemini model to create outfits from a user's wishlist.",
    version="1.0.0"
)

# --- Pydantic Models for Input and Output Validation ---

# Input Models (from your original code, no changes needed)
class WishlistItem(BaseModel):
    name: str
    description: str
    productId: str

class Wishlist(BaseModel):
    items: list[WishlistItem]

# Output Models (NEW: for response validation and documentation)
class OutfitItem(BaseModel):
    name: str
    productId: str

class Outfit(BaseModel):
    outfitId: str
    items: list[OutfitItem]

class OutfitsResponse(BaseModel):
    outfits: list[Outfit]


# --- Core Logic ---

async def ask_gemini(wishlist_items: list[WishlistItem]) -> str:
    """
    Constructs a detailed prompt and sends it to the Gemini API.
    """
    # Convert Pydantic models to a simple dict list for the prompt
    wishlist_json_str = json.dumps([item.dict() for item in wishlist_items], indent=2)

    # --- IMPROVED PROMPT ---
    prompt = f"""
You are an expert fashion stylist AI assistant. Your sole purpose is to create stylish and coherent outfits from a given list of clothing items.

You MUST follow these rules strictly:
1.  Analyze the provided "Wishlist" JSON data.
2.  Create one or more complete outfits by combining items that complement each other.
3.  Your response MUST BE a valid JSON array of outfit objects and nothing else.
4.  Do NOT include any explanatory text, comments, markdown formatting (like ```json), or any characters before or after the JSON array.

The output JSON array must conform to the following structure:
[
  {{
    "outfitId": "A unique string identifier for the outfit (e.g., 'outfit_1').",
    "items": [
      {{
        "name": "The name of the item from the wishlist.",
        "productId": "The corresponding productId from the wishlist."
      }}
    ]
  }}
]

Here is an example:
---
Wishlist:
[
  {{
    "name": "Classic Black T-shirt",
    "description": "100% cotton crewneck",
    "productId": "B-TS-001"
  }},
  {{
    "name": "Slim-fit Chinos",
    "description": "Beige color, stretch fabric",
    "productId": "P-CH-005"
  }}
]

Your Response:
[
  {{
    "outfitId": "outfit_1",
    "items": [
      {{
        "name": "Classic Black T-shirt",
        "productId": "B-TS-001"
      }},
      {{
        "name": "Slim-fit Chinos",
        "productId": "P-CH-005"
      }}
    ]
  }}
]
---

Now, create outfits based on this wishlist:

Wishlist:
{wishlist_json_str}

Your Response:
"""

    response =  client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {"role": "system", "content": "You are a JSON-only fashion stylist AI."},
            {"role": "user", "content": prompt}
        ],
        # Using response_format with compatible models can improve reliability
        # response_format={"type": "json_object"}, # Note: Check if the Google API via OpenAI proxy supports this. If not, the strong prompt is your best bet.
        temperature=0.5, # Lower temperature for more predictable, structured output
    )
    
    if not response.choices:
        raise HTTPException(status_code=500, detail="Gemini API returned an empty response.")
        
    return response.choices[0].message.content


# --- API Endpoint ---

@app.post("/build-outfits", response_model=OutfitsResponse)
async def build_outfits(wishlist: Wishlist):
    """
    Accepts a wishlist of clothing items and returns curated outfits.
    """
    try:
        # 1. Get raw string response from the model
        outfits_raw_str = await ask_gemini(wishlist.items)

        # 2. Parse the JSON string into Python objects
        # The model might still wrap the JSON in markdown, so we can try to strip it
        clean_json_str = outfits_raw_str.strip().removeprefix("```json").removesuffix("```")
        outfits_data = json.loads(clean_json_str)

        # 3. Validate the data structure using our Pydantic response model
        validated_response = OutfitsResponse(outfits=outfits_data)
        
        return validated_response

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500, 
            detail="Failed to decode JSON from Gemini's response."
        )
    except ValidationError as e:
        # This is triggered if the JSON is valid but doesn't match our OutfitsResponse model
        raise HTTPException(
            status_code=500, 
            detail=f"Gemini's JSON output does not match the required format: {e}"
        )
    except Exception as e:
        # Catch any other potential errors (e.g., API connection issues)
        raise HTTPException(status_code=500, detail=str(e))