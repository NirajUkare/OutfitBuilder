from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import uvicorn

# --- Configuration ---
load_dotenv()
# It's good practice to validate that environment variables are set
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# Configure the Gemini API
genai.configure(api_key=gemini_api_key)

# Make the model name configurable
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # Using gemini-pro as default

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

# Output Models (for response validation and documentation)
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

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {str(e)}")


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
    
if __name__ == "__main__":
    # Get the PORT from the environment variable provided by the platform (e.g., Render)
    port = int(os.environ.get("PORT", 8000))
    # '0.0.0.0' is the host that makes the app accessible from outside the container
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)