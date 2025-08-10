from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List
import os, requests, fitz, tempfile, io, asyncio, time, re, json
from PIL import Image
import pytesseract
import httpx
from dotenv import load_dotenv
from bs4 import BeautifulSoup

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


REQUESTS_SESSION = requests.Session()
REQUESTS_SESSION.headers.update({"Accept-Encoding": "gzip, deflate"})
REQUESTS_SESSION.mount(
    "https://",
    HTTPAdapter(
        pool_connections=20, pool_maxsize=20,
        max_retries=Retry(total=2, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504])
    ),
)

load_dotenv()

#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_TOKEN = os.getenv("API_TOKEN")

app = FastAPI()

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

FULL_PROMPT_TEMPLATE = """You are an insurance policy expert. Use ONLY the information provided in the context to answer the questions.
Context:
{context}

Questions:
{query}

Instructions:
1. Provide clear and direct answers based ONLY on the context.
2. Do not specify the clause number or clause description.
3. If the answer is "Yes" or "No," include a short explanation.
4. If question is not related to the context, reply: "Not mentioned in the policy."
5. Give each answer in a single paragraph without numbering.

Answers:"""

CHUNK_PROMPT_TEMPLATE = """You are an insurance policy expert. Use ONLY the chunked context and questions to answer.
Context:
{context}

Questions:
{query}

Instructions:
1. Combine insights from all chunks.
2. Don’t repeat content.
3. If answer not found and the question is not related to the context then, reply: "Not mentioned in the policy."
4. Be concise (max one paragraph/answer).
5. Give each answer in a single paragraph without numbering.
6.If answer not found but related to context, answer from web source.

Answers:"""

WEB_PROMPT_TEMPLATE = """You are an expert insurance policy assistant. Based on the document titled "{title}", answer the following questions using general or public insurance knowledge.
Title: "{title}"

Questions:
{query}

Instructions:
- Use public knowledge.
- If specific document data is needed, reply: "Not mentioned in the policy."
- Keep each answer concise (1 paragraph max).
- Give each answer in a single paragraph without numbering.
- If question is not related to topic reply with, "Not mentioned in the policy".

Answers:"""

PUZZLE_PROMPT_TEMPLATE = """You are a puzzle-solving expert.
You are given the full text of a PDF and the outputs from any instructions that were executed from the PDF.
Use BOTH the PDF content and the executed instruction outputs to solve the puzzle hidden inside.

PDF Content:
{pdf_text}

Executed Instruction Results:
{instruction_results}

Task:
- Solve the puzzle and return ONLY the final answer (no explanation, no steps, no reasoning).
- The output must be a single sentence.

FINAL ANSWER:"""

# ---------------- Instruction Detection ----------------
INSTRUCTION_RX = re.compile(
    r'\b(POST|GET|PUT|DELETE)\s+(https?://[^\s]+)(?:\s*({.*?}))?',
    re.IGNORECASE | re.DOTALL
)

def find_and_execute_instructions(text: str):
    results = []
    for match in INSTRUCTION_RX.finditer(text):
        method = match.group(1).upper()
        url = match.group(2).strip()
        body_raw = match.group(3)

        try:
            if method == "GET":
                resp = REQUESTS_SESSION.get(url, timeout=15)
            elif method in {"POST", "PUT", "DELETE"}:
                if body_raw:
                    try:
                        body_json = json.loads(body_raw)
                        resp = REQUESTS_SESSION.request(method, url, json=body_json, timeout=15)
                    except Exception:
                        resp = REQUESTS_SESSION.request(method, url, data=body_raw, timeout=15)
                else:
                    resp = REQUESTS_SESSION.request(method, url, timeout=15)
            else:
                continue

            try:
                content = resp.json()
            except Exception:
                content = resp.text[:1000]

            results.append({
                "method": method,
                "url": url,
                "status": resp.status_code,
                "response": content
            })
        except Exception as e:
            results.append({
                "method": method,
                "url": url,
                "error": str(e)
            })
    return results




# Extract text from PDF (OCR if needed)
def extract_text_from_pdf_url(pdf_url: str) -> tuple[str, int, str]:
    response = requests.get(pdf_url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    full_text = ""
    title = ""
    with fitz.open(tmp_path) as doc:
        page_count = len(doc)

        # Title extraction
        for i in range(min(15, page_count)):
            page = doc[i]
            text = page.get_text().strip()

            if not text:
                try:
                    pix = page.get_pixmap(dpi=100)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    text = pytesseract.image_to_string(img, lang="eng").strip()
                except:
                    continue

            if text:
                title = text.split('\n')[0][:100]
                break

        # Full extraction only if ≤ 200 pages
        if page_count <= 200:
            for i in range(page_count):
                page = doc[i]
                text = page.get_text().strip()

                if not text:
                    try:
                        pix = page.get_pixmap(dpi=100)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        text = pytesseract.image_to_string(img, lang="eng").strip()
                    except:
                        continue

                full_text += text + "\n"

    os.remove(tmp_path)
    return (full_text.strip() if page_count <= 200 else "", page_count, title or "Untitled Document")

def extract_text_from_web_url(web_url: str) -> tuple[str, int, str]:
    """
    Fetch HTML content from a webpage and extract readable text.
    Returns (full_text, page_count_placeholder, title)
    page_count is 1 for web content to fit into parameter choices
    """
    r = REQUESTS_SESSION.get(web_url, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Title
    title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled Webpage"

    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    # Extract visible text and try to collapse redundant whitespace
    text = "\n".join(line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip())

    return (text.strip(), 1, title)

def split_text(text: str, chunk_size=1500, overlap=200) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks[:10]

# Mistral LLM
def call_mistral(prompt: str) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-small-latest",
        "temperature": 0.7,
        "top_p": 1,
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}]
    }
    res = requests.post(url, headers=headers, json=payload, timeout=15)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

def call_grok(prompt: str)->str:
    url="https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "temperature": 0.3,
            "top_p": 1,
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        }
    res = requests.post(url, headers=headers, json=payload, timeout=15)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

def call_mistral_on_chunks(chunks: List[str], questions: List[str]) -> List[str]:
    question_block = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    combined_context = "\n\n".join(chunks)
    prompt = CHUNK_PROMPT_TEMPLATE.format(context=combined_context, query=question_block)
    answer = call_mistral(prompt)
    return [a.strip() for a in answer.split("\n") if a.strip()]


# Groq fallback
async def call_groq_on_chunks(chunks: List[str], questions: List[str]) -> List[str]:
    question_block = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    async def call_chunk(chunk):
        prompt = CHUNK_PROMPT_TEMPLATE.format(context=chunk, query=question_block)
        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "temperature": 0.3,
            "top_p": 1,
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        }
        async with httpx.AsyncClient() as client:
            res = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]

    responses = await asyncio.gather(*[call_chunk(chunk) for chunk in chunks])
    all_answers = "\n".join(responses)
    return [a.strip() for a in all_answers.split("\n") if a.strip()]

def solve_puzzle_with_llm(pdf_text: str, instruction_results: list, page_count: int) -> str:
    """
    Send full pdf_text + instruction_results to LLM and return only the final answer.
    Uses Mistral first, Groq as async fallback.
    """
    combined_results = json.dumps(instruction_results, indent=2) if instruction_results else "No instructions executed."
    # Keep prompts within reasonable size -> but user asked to send "complete text" only for puzzle type.
    # We'll still cap to avoid request failure, but you can increase if your model/context supports it.
    pdf_for_prompt = pdf_text if len(pdf_text) < 300000 else pdf_text[:300000]  # ~300k chars cap (adjust as needed)
    prompt = PUZZLE_PROMPT_TEMPLATE.format(pdf_text=pdf_for_prompt, instruction_results=combined_results)

    # choose params
    try:
        resp = call_grok(prompt).strip()
        # Ensure we return only the first non-empty line (final answer)
        for line in resp.splitlines():
            if line.strip():
                return line.strip()
        return resp.strip()
    except Exception:
        try:
            loop = asyncio.get_event_loop()
            resp = loop.run_until_complete(call_mistral(prompt))
            for line in resp.splitlines():
                if line.strip():
                    return line.strip()
            return resp.strip()
        except Exception:
            return "LLM_ERROR: Unable to get puzzle answer."



@app.get("/")
def read_root():
    return {"message": "PDF API is running"}

# Main Endpoint
@app.post("/api/v1/hackrx/run")
async def run_analysis(request: RunRequest, authorization: str = Header(...)):
    print(request)
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        start_time = time.time()

        is_pdf = False
        try:
            head = REQUESTS_SESSION.head(request.documents, allow_redirects=True, timeout=8)
            ctype = head.headers.get("content-type", "").lower()
            if "application/pdf" in ctype:
                is_pdf = True
        except Exception:
            # if HEAD fails, fall back to extension detection
            if request.documents.lower().endswith(".pdf"):
                is_pdf = True

        if is_pdf:
            full_text, page_count, title = extract_text_from_pdf_url(request.documents)
            is_web = False
        else:
            full_text, page_count, title = extract_text_from_web_url(request.documents)
            is_web = True



        # full_text, page_count, title = extract_text_from_pdf_url(request.documents)
        chunks = split_text(full_text) if full_text else []
        question_block = "\n".join([f"{i+1}. {q}" for i, q in enumerate(request.questions)])

        instruction_results = find_and_execute_instructions(full_text) if full_text else []
        puzzle_answer = None
        if instruction_results:
            # keep existing behavior: we may still solve puzzle but do not return only final answer by default
            puzzle_answer = solve_puzzle_with_llm(full_text, instruction_results, page_count)
            return {"answer": puzzle_answer}


        if is_web:
            try:
                prompt = FULL_PROMPT_TEMPLATE.format(context=full_text, query=question_block)
                resp = call_mistral(prompt)
                answers = [re.sub(r"^\d+[\.\)]\s*", "", a.strip()) for a in resp.split("\n") if a.strip()]
                return {"answers": answers}
            except Exception:
                try:
                    answers = await call_groq_on_chunks([full_text], request.questions)
                    return {"answers": answers}
                except Exception:
                    raise HTTPException(status_code=500, detail="Both LLMs failed for webpage content.")



        # Case 1: Full PDF (<= 100 pages)
        if page_count <= 100:
            try:
                prompt = FULL_PROMPT_TEMPLATE.format(context=full_text, query=question_block)
                response = call_mistral(prompt)
                answers = [re.sub(r"^\d+[\.\)]\s*", "", a.strip()) for a in response.split("\n") if a.strip()]
                return {"answers": answers}
            except Exception:
                raise HTTPException(status_code=500, detail="Mistral failed on full content.")

        # Case 2: Chunked PDF (<= 200 pages)
        elif page_count <= 200:
            try:
                answers = call_mistral_on_chunks(chunks, request.questions)
                return {"answers": answers}
            except:
                try:
                    answers = await call_groq_on_chunks(chunks, request.questions)
                    return {"answers": answers}
                except:
                    raise HTTPException(status_code=500, detail="All LLMs failed for chunks.")

        # Case 3: Large (> 200 pages) → Use title and public info
        else:
            try:
                prompt = WEB_PROMPT_TEMPLATE.format(title=title, query=question_block)
                response = call_mistral(prompt)
                answers = [re.sub(r"^\d+[\.\)]\s*", "", a.strip()) for a in response.split("\n") if a.strip()]
                return {"answers": answers}
            except:
                try:
                    answers = await call_groq_on_chunks([title], request.questions)
                    return {"answers": answers}
                except:
                    raise HTTPException(status_code=500, detail="All LLMs failed for >200 page fallback.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        print(f"⏱️ Total processing time: {round(time.time() - start_time, 2)}s")
