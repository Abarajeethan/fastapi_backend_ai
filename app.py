import os, base64, requests, json
from io import BytesIO
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from PIL import Image

# ───────────────────────────────
# Load API keys
# ───────────────────────────────
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not OPENAI_KEY:
    raise Exception("❌ Missing OPENAI_API_KEY in .env")

client_openai = OpenAI(api_key=OPENAI_KEY)
#client_gemini = genai.Client(api_key=GEMINI_KEY)

# ───────────────────────────────
# FastAPI setup
# ───────────────────────────────
app = FastAPI(title="STEP AI Lifestyle Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ───────────────────────────────
# Data models
# ───────────────────────────────
class AnalyzeRequest(BaseModel):
    image_url: str

class AIImageRequest(BaseModel):
    fileUrl: str
    prompt: str
    ai_model: str = "OPENAI"  # "OPENAI" or "GEMINI"


# ───────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "message": "STEP AI backend running ✅"}


# ───────────────────────────────
# 1️⃣ Analyze image → Generate prompt
# ───────────────────────────────
@app.post("/analyze-image")
def analyze_image(req: AnalyzeRequest):
    system_prompt = """
    You are an expert creative director and AI assistant for Stockmann.

    - Premium Nordic aesthetic
    - Soft natural light (morning / golden hour)
    - Realistic human realism (no cartoon)
    - Product is the hero and unchanged

    Output JSON:
    {
    "product_category": "A breadcrumb-style category path. Example: 'Home > Kitchen > Tableware > Mugs'",
    "main_product_description": "A concise, 1-sentence description of the main product. Example: 'A white porcelain coffee mug with a gold handle.'",
    "key_attributes": [
    "An array of 3-5 keywords describing the product's style, material, and color. Example: ['porcelain', 'white', 'gold', 'minimalist', 'luxury']"
    ],
    "image_type": "Classify as 'supplier_white_background', 'supplier_lifestyle', or 'photoshoot_with_model'.",
    "generated_editing_prompt": "This is the new prompt for the image editor. See instructions below."
    }

    **Instructions for `generated_editing_prompt`:**
    * Based on the `product_category` and `key_attributes`, create a concise, descriptive prompt for an image editing model.
    * This prompt *must* adhere to the **Stockmann Brand Guide**.
    * The prompt must describe a new, *logically appropriate* background or setting for the product.
    * **Contextual Scene Logic:**
    * **Home & Decor:** Products must be in the correct room (e.g., sofas in a living room, knives in a kitchen, bedding in a bedroom). The setting should be a stylish, minimalist Nordic interior.
    * **Outdoor/Active Gear:** (e.g., backpacks, hiking boots, functional jackets). Place these items in an appropriate, clean, and aspirational outdoor setting (e.g., a city park, a serene forest trail, a clean urban street). If it's an item worn or carried (like a backpack), the prompt should place it *on a
    person* in that setting.
    * **Seasonal/Weather Apparel:** (e.g., winter coats, rain jackets, swimwear). The setting *must* match the item's function. A winter coat should be in a clean, snowy urban or natural landscape. A raincoat could be on a person in a misty, post-rain Helsinki street.
    * **Fashion (General/Occasion):** (e.g., dresses, suits, casual wear). Place in an aspirational, on-brand setting, such as a modern apartment, an architectural space, or a minimalist studio.
    * **If `image_type` is 'photoshoot_with_model':** Your first priority is to keep the model and their outfit. The prompt should replace the *existing* background with a simpler, on-brand studio, abstract, or architectural setting that complements the fashion.
    * **Crucially:** The prompt must *start with a clear instruction to identify the main product (or model and product), keep it unchanged, and replace only its background*.
    * The prompt must *end with a specific instruction* on lighting and realism (e.g., "photorealistic with soft, natural morning light").
    **Examples for `generated_editing_prompt`:**
    * **(For a Kitchen Knife):** "Identify the chef's knife in this image and keep it identical. Replace its background with a dark, minimalist marble kitchen counter. Add a sprig of fresh basil and a few cherry tomatoes nearby. The scene must be photorealistic with warm, side-lit lighting."
    * **(For a Sofa):** "Identify the grey linen sofa and keep it perfectly unchanged. Place it in a minimalist Scandinavian living room with a light wooden floor, a white shag rug, and a large, healthy monstera plant in the corner. The scene must be photorealistic with bright, indirect sunlight from a
    window."
    * **(For a Backpack on white background):** "Identify the grey canvas backpack and keep it unchanged. Place it on a person walking down a clean, tree-lined street in Helsinki in autumn. The scene must be photorealistic with crisp, natural daylight."
    * **(For a Model in a Winter Coat):** "Identify the model and their dark wool winter coat, keeping them unchanged. Replace the background with a minimalist, snowy park scene during a soft 'golden hour' glow. The scene must be photorealistic."
    * **(For a Model in a Dress):** "Identify the model and their outfit in this image and keep them unchanged. Replace the current background with an abstract, soft-focus studio setting in a warm, neutral beige tone. The scene must be photorealistic with professional, soft-box lighting."
        """

    try:
        r = requests.get(req.image_url, timeout=15)
        r.raise_for_status()
        b64_img = base64.b64encode(r.content).decode("utf-8")
    except Exception as e:
        return {"success": False, "error": f"Failed to download image: {e}"}

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "Analyze this image and output the JSON:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
        ]}
    ]

    try:
        resp = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"}
        )
        result = json.loads(resp.choices[0].message.content)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ───────────────────────────────
# 2️⃣ Generate lifestyle image (OpenAI / Gemini)
# ───────────────────────────────
@app.post("/ai-image")
def ai_image(req: AIImageRequest):
    try:
        r = requests.get(req.fileUrl, timeout=20)
        r.raise_for_status()
        img_bytes = r.content
        Image.open(BytesIO(img_bytes))  # validate

        # ─────────── DALL·E 3 ───────────
        if req.ai_model.upper() == "OPENAI":
            prompt_text = (
                f"Identify the main product in this image and keep it identical. "
                f"Replace only the background as described: {req.prompt}. "
                f"The result must be photorealistic, Nordic natural light."
            )

            files = {
                "model": (None, "gpt-image-1"),
                "prompt": (None, prompt_text),
                "image": ("image.jpg", img_bytes, "image/jpeg"),
                "size": (None, "1024x1024")
            }

            r = requests.post(
                "https://api.openai.com/v1/images/edits",
                headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                files=files,
                timeout=120
            )
            data = r.json()

            if "error" in data:
                return {"success": False, "error": data["error"]["message"]}

            image_info = data.get("data", [{}])[0]
            if "url" in image_info:
                return {"success": True, "source": req.fileUrl, "image_url": image_info["url"]}
            elif "b64_json" in image_info:
                return {"success": True, "source": req.fileUrl, "image_base64": image_info["b64_json"]}
            else:
                return {"success": False, "error": "No image data returned from DALL-E."}

        else:
            return {"success": False, "error": f"Unknown model '{req.ai_model}'"}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ───────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")
