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
      "product_category": "...",
      "main_product_description": "...",
      "key_attributes": ["..."],
      "image_type": "supplier_white_background | supplier_lifestyle | photoshoot_with_model",
      "generated_editing_prompt": "Prompt following brand guide"
    }
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
