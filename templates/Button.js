import os
import json
import numpy as np
from flask import Flask, render_template, request, send_file, session
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import re
from markdown import markdown as md
from langdetect import detect
from deep_translator import GoogleTranslator
import io

# 🔐 Load API key and Flask app setup
load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ['secure_random_key']

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 📁 Config
TEXT_DIR = "text_files"
CHUNK_SIZE_TOKENS = 300
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4"
chat_memory_path = "chat_memory.json"
entity_memory_path = "entity_memory.json"
encoder = tiktoken.get_encoding("cl100k_base")

# 📦 Stores
filename_embeddings = {}
file_chunks = {}
chunk_embeddings = {}
section_lookup = {}

# 🧠 Memory functions
def load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed loading {path}: {e}")
    return default

def save_json(data, path):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed saving {path}: {e}")

chat_memory = load_json(chat_memory_path, {"conversations": []})
entity_memory = load_json(entity_memory_path, {})

# Nieuwe data structuur in chat_memory:
# "conversations": [ { "user": str, "assistant": str, "feedback": "up"/"down"/None, "score": int, "embedding": list[float] } ]

# 🔤 Query rewriting with synonyms
SYNONYMS = {
    "open": "established",
    "opened": "established",
    "founding": "established",
    "founded": "established",
    "principal": "Head of School",
    "headmaster": "Head of School",
    "director": "Head of School",
    "students": "enrollment",
    "student body": "enrollment",
    "teachers": "faculty",
    "staff": "faculty",
    "overview": "overview",
    "about": "overview"
}
def rewrite_question(question):
    q = question.lower()
    for word, repl in SYNONYMS.items():
        q = q.replace(word, repl)
    return q

# ✂️ Chunk text into token-limited chunks with section headers
def chunk_text(text, filename):
    chunks, current, section = [], [], "Unknown Section"
    lines = text.splitlines()
    for line in lines:
        if re.match(r"^#+\s", line):
            section = line.strip("# ").strip()
        words = line.split()
        for word in words:
            current.append(word)
            if len(encoder.encode(" ".join(current))) >= CHUNK_SIZE_TOKENS:
                chunk = f"[File: {filename}]\n[Section: {section}]\n{' '.join(current)}"
                chunks.append((section, chunk))
                current = []
    if current:
        chunk = f"[File: {filename}]\n[Section: {section}]\n{' '.join(current)}"
        chunks.append((section, chunk))
    return chunks

# 🔎 Get embeddings from OpenAI
def get_embedding(text):
    try:
        embedding = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
        return np.array(embedding.data[0].embedding)
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        return np.zeros(1536)

# 📚 Load all files, chunk, embed filenames & chunks
def load_files():
    print("[INFO] Loading and embedding text files...")
    for filename in os.listdir(TEXT_DIR):
        if filename.endswith(".txt"):
            try:
                with open(os.path.join(TEXT_DIR, filename), "r", encoding="utf-8") as f:
                    chunks = chunk_text(f.read(), filename)
                    file_chunks[filename] = chunks
                    filename_embeddings[filename] = get_embedding(filename)
                    for i, (section, chunk) in enumerate(chunks):
                        chunk_embeddings[(filename, i)] = get_embedding(chunk)
                        section_lookup.setdefault(section, []).append((filename, i))
            except Exception as e:
                print(f"[ERROR] Failed processing {filename}: {e}")
    print("[INFO] Finished loading files.")

# 📂 Find best matching files by cosine similarity
def get_best_files(user_input, top_n=3):
    input_vec = get_embedding(user_input)
    scored = [(cosine_similarity([input_vec], [vec])[0][0], fname)
              for fname, vec in filename_embeddings.items()]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [fname for _, fname in scored[:top_n]]

# 🔍 Select top chunks from best files with weighting & token budget
def get_top_chunks(user_input, files, max_token_budget=3000):
    input_vec = get_embedding(user_input)
    section_scores = {}
    for fname in files:
        for i, (section, chunk) in enumerate(file_chunks[fname]):
            vec = chunk_embeddings[(fname, i)]
            score = cosine_similarity([input_vec], [vec])[0][0]
            token_count = len(encoder.encode(chunk))
            weight = 1.2 if any(k in section.lower() for k in ["overview", "about"]) else 1.0
            section_scores.setdefault(section, []).append((score * weight, chunk, fname, token_count))
    all_chunks = [item for sub in section_scores.values() for item in sub]
    all_chunks.sort(key=lambda x: x[0], reverse=True)
    total = 0
    selected = []
    for score, chunk, fname, tokens in all_chunks:
        if total + tokens <= max_token_budget:
            selected.append((score, chunk, fname))
            total += tokens
    return selected

# 🔧 Haal de meest relevante feedback-voorbeelden (good en bad) op uit chat_memory
def get_relevant_feedback_examples(user_vec, top_k=3):
    good_examples = []
    bad_examples = []
    for conv in chat_memory.get("conversations", []):
        if "feedback" in conv and conv["user"] and conv["assistant"] and "embedding" in conv and conv["embedding"]:
            emb = np.array(conv["embedding"])
            similarity = cosine_similarity([user_vec], [emb])[0][0]
            score = conv.get("score", 0)
            weighted_sim = similarity * score
            if conv["feedback"] == "up" and score > 0 and weighted_sim > 0.7:
                good_examples.append((weighted_sim, conv))
            elif conv["feedback"] == "down" and score < 0 and weighted_sim > 0.7:
                bad_examples.append((weighted_sim, conv))
    good_examples.sort(key=lambda x: x[0], reverse=True)
    bad_examples.sort(key=lambda x: x[0], reverse=True)
    return [ex[1] for ex in good_examples[:top_k]], [ex[1] for ex in bad_examples[:top_k]]

def ask_reasoning_engine(user_input, context_chunks, history_limit=5):
    try:
        if not user_input.strip():
            return "⚠️ Lege vraag ontvangen."

        # Detect language
        try:
            detected_lang = detect(user_input)
        except:
            detected_lang = "en"

        # Translate to English
        try:
            translated_input = GoogleTranslator(source='auto', target='en').translate(user_input)
        except:
            translated_input = user_input

        user_vec = get_embedding(user_input)

        # Recent conversations (history)
        recent_conversations = chat_memory.get("conversations", [])[-history_limit:]
        history_context = "\n\n".join(
            f"User: {c['user']}\nAI: {c['assistant']}"
            for c in recent_conversations if c['user'] and c['assistant']
        )

        # Context from docs
        context = "\n\n".join([
            f"From {fname} (likely section: {chunk.split('[Section: ')[-1].split(']')[0]}):\n{chunk}"
            for _, chunk, fname in context_chunks
        ])

        # Entity memory
        entities = "\n".join([f"{k}: {v}" for k, v in entity_memory.items()])

        # Get relevant good/bad feedback examples based on embedding similarity & score
        good_examples, bad_examples = get_relevant_feedback_examples(user_vec)

        good_section = "\n".join(
            f"Q: {ex['user']}\nA: {ex['assistant']}" for ex in good_examples
        ) or "No good examples yet."

        bad_section = "\n".join(
            f"Q: {ex['user']}\nA: {ex['assistant']}" for ex in bad_examples
        ) or "No bad examples yet."

        # System prompt with dynamic good/bad examples
        prompt = f"""
### 🧠 You are a highly intelligent AI assistant designed to emulate GPT-4 reasoning behavior.

You synthesize accurate, clear, and well-organized answers using:
- Uploaded document chunks (with metadata)
- User conversation history
- Your own inference capabilities

---

### 💼 Your Role:
- Emulate a **knowledgeable expert** who prioritizes **precision** and **clarity**
- Use document content when it’s **relevant and helpful**
- Maintain **human-like tone** and **logical reasoning**

---

### 📝 Response Formatting Guidelines:
- Use `###` for section headers
- Use **bold** for important terms or facts
- Use *italics* for nuance
- Use bullet points or numbered lists when helpful
- Start each answer with a brief summary sentence or direct answer

---
### ❗ Instructions:
1. **Never say "I don’t know"**. Instead, infer a likely answer or explain why data is missing.
2. **Do not guess wildly**—be thoughtful, clear, and explain your reasoning.
3. Favor **specific** and **relevant** information over vague generalities.
4. When multiple possible answers exist, explain them and **recommend** the most likely.

### ✅ Preferred Answer Styles (from past upvotes):
{good_section}

### ⚠️ Avoid These Styles (from past downvotes):
{bad_section}

---
### Entity Memory:
{entities}

### Document Context:
{context}

### Chat History:
{history_context}
"""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": translated_input}
        ]

        result = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
        answer = result.choices[0].message.content.strip()

        # Update entity memory with new facts if any
        for line in answer.split("\n"):
            if ":" in line and len(line.split(":")) == 2:
                key, val = line.split(":")
                if len(key) < 30:
                    entity_memory[key.strip()] = val.strip()
                    save_json(entity_memory, entity_memory_path)

        # Translate answer back if needed
        translated_answer = GoogleTranslator(source='en', target=detected_lang).translate(answer) if detected_lang != 'en' else answer

        print("[DEBUG] Answer generated:")
        print(translated_answer)

        return md(translated_answer)

    except Exception as e:
        print(f"[ERROR] Failed to generate answer: {e}")
        return f"❌ Er is een fout opgetreden: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    user_input = ""

    if request.method == "POST":
        user_input = request.form["user_input"]
        rewritten = rewrite_question(user_input)
        best_files = get_best_files(rewritten)
        top_chunks = get_top_chunks(rewritten, best_files)
        answer = ask_reasoning_engine(user_input, top_chunks)
        # Save conversation with default no feedback and score 0
        embedding = get_embedding(user_input).tolist()
        chat_memory["conversations"].append({
            "user": user_input,
            "assistant": answer,
            "feedback": None,
            "score": 0,
            "embedding": embedding
        })
        save_json(chat_memory, chat_memory_path)
        session['last_response'] = answer
        session['last_user_input'] = user_input

    return render_template("index.html", response=answer, user_input=user_input)

@app.route("/download")
def download():
    response_text = session.get('last_response')
    if not response_text:
        return "No AI response available to download.", 404

    buffer = io.BytesIO()
    buffer.write(response_text.encode('utf-8'))
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="ai_response.txt",
        mimetype="text/plain"
    )

# 👍👎 Feedback route
@app.route("/feedback", methods=["POST"])
def feedback():
    user_input = session.get("last_user_input", "")
    fb = request.form.get("feedback")
    last_response = session.get("last_response", "")

    print(f"[FEEDBACK] User gave a {fb} for: {user_input}")

    # Find last conversation matching this input + answer
    for conv in reversed(chat_memory["conversations"]):
        if conv["user"] == user_input and conv["assistant"] == last_response and conv["feedback"] is None:
            conv["feedback"] = fb
            conv["score"] = 1 if fb == "up" else -1
            break

    save_json(chat_memory, chat_memory_path)

    if fb == "down":
        # Force AI to improve answer immediately
        rewritten = rewrite_question(user_input)
        best_files = get_best_files(rewritten)
        top_chunks = get_top_chunks(rewritten, best_files)
        improved_answer = ask_reasoning_engine(
            user_input + "\n\nPlease improve your answer slightly: be clearer, more helpful, or structured.",
            top_chunks
        )
        session['last_response'] = improved_answer
        return render_template("index.html", response=improved_answer, user_input=user_input)

    return render_template("index.html", response=last_response, user_input=user_input)

if __name__ == "__main__":
    load_files()
    app.run(debug=True, host="0.0.0.0", port=3000)
