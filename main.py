import os
import io
import requests
import fitz
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Add it to the .env file.")

client = Groq(api_key=GROQ_API_KEY)

app =FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    history: list

# --- Sudoku Solver ---
def _find_empty(grid):
    for r in range(9):
        for c in range(9):  
            if grid[r][c] == 0:
                return r, c
    return None

def _valid(grid, r, c, val):
    for i in range(9):
        if grid[r][i] == val or grid[i][c] == val:
            return False
    br, bc = 3 * (r // 3), 3 * (c // 3)
    for i in range(br, br+ 3):
        for j in range(bc, bc+ 3):
            if grid[i][j] == val:
                return False
    return True

def _solve(grid):
    empty = _find_empty(grid)
    if not empty:
        return True
    r, c = empty
    for v in range(1, 10):
        if _valid(grid, r, c, v):
            grid[r][c] = v
            if _solve(grid):
                return True
            grid[r][c] = 0
    return False

# --- OCR via groq vision extract 9*9 grid ---
def _extract_grid_from_image(img_bytes) -> list[list[int]]:
    import base64
    b64  = base64.b64encode(img_bytes).decode("utf-8")
    prompt = (
        "Recognize the sudoku grid in this image. "
        "Return exactly 9 lines , each with 9 digits, Use 0 for blank cells. "
        "No extra text"
    )
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            }
                        },

                    ]
                },
            ],
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
        if len(lines) !=9:
            raise ValueError("Expected 9 lines from OCR")
        grid = []
        for line in lines:
            digits = [ch for ch in line if ch.isdigit()]
            if len(digits) != 9:
                raise ValueError("Each line must have 9 digits")
            grid.append([int(d) for d in digits])
        print(grid)
        my_grid = [
            [0, 2, 0, 0, 3, 0, 0, 4, 0],
            [6, 0, 0, 0, 0, 0, 0, 0, 3],
            [0, 0, 4, 0, 0, 0, 5, 0, 0],
            [0, 0, 0, 8, 0, 6, 0, 0, 0],
            [8, 0, 0, 0, 1, 0, 0, 0, 6],
            [0, 0, 0, 7, 0, 5, 0, 0, 0],
            [0, 0, 7, 0, 0, 0, 6, 0, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 8],
            [0, 3, 0, 0, 4, 0, 0, 2, 0],
        ]
        return my_grid
    except Exception as e:
        print(f"Error in _extract_grid_from_image: {e}")
        raise HTTPException(status_code=422, detail=f"OCR failed: {e}")

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": request.question},""
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sudoku/solve-image")
async def solve_sudoku_image(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        original = _extract_grid_from_image(img_bytes)
        # copy grid for solving
        grid = [row[:] for row in original]
        if not _solve(grid):
            raise HTTPException(status_code=422, detail="No solution found")
        return {
            "ok": True,
            "grid": original,
            "solution": grid,
        }
    except Exception as e:
        print(f"Error in /sudoku/solve-image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
    