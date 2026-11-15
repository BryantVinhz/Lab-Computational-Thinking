# 🧭 Lumen Travel — Sign-in Travel Assistant with RAG Chat (Zurich)

A Retrieval-Augmented Generation (RAG) travel assistant built with **Streamlit**, **Firebase Authentication**, and **Ollama** (`embeddinggemma` + `qwen3:4b`). The assistant answers traveler questions about **Zurich, Switzerland** strictly using a curated local travel guide (`knowledge_base.json`), cites supporting guide IDs, and refuses to guess when information is missing.

---

## I. Student Information & Demo

* **Full Name:** `Trần Vĩnh Kiệt`
* **Student ID:** `25127399`
* **Course:** CSC10014 — Computational Thinking (University of Science, VNU-HCM)
* **Chosen City:** Zurich, Switzerland
* **Demonstration Video (Max 5 mins):** [https://drive.google.com/drive/folders/1P1bV3N82eOMSyZQnwreRKGGoLPFcbVLK?usp=sharing](https://drive.google.com/drive/folders/1P1bV3N82eOMSyZQnwreRKGGoLPFcbVLK?usp=sharing)

---

## II. Key Features

1. **Firebase Email/Password Authentication (`firebase_client.py`):**
   * Supports user sign-up and sign-in via Firebase Auth REST API.
   * Protects the RAG chat interface so only authenticated users can access it.
2. **Curated Travel Knowledge Base (`knowledge_base.json`):**
   * Contains **20 concise entries (`G01`–`G20`)** covering Zurich landmarks, museums, parks, markets, nightlife, local transit (`ZVV`), airport transfer (`ZRH`), and cultural etiquette.
   * Place entries include `category` and `tags` metadata prepared for recommendation filtering in `RAG-Rec` problem.
3. **One-Time Vector Indexing & Semantic Retrieval (`rag.py`):**
   * Embeds all guide entries once using `@st.cache_resource` and `embeddinggemma`.
   * Retrieves the top-$k$ ($k=3$) most relevant entries via cosine similarity:
     $$\text{cosine}(q, d) = \frac{q \cdot d}{\|q\|\|d\|}$$
4. **Grounded Generation & Hallucination Control (`rag.py`):**
   * Generates concise answers using `qwen3:4b` (`temperature=0`) strictly from retrieved passages, citing IDs such as `[G08]`.
   * Explicitly responds with `"I don't know"` when the guide lacks supporting evidence.
   * Displays retrieved entries and cosine similarity scores in an expandable UI panel (`st.expander`).
5. **Ephemeral Session Memory (`app.py`):**
   * Stores chat history in `st.session_state` for the active session and clears it completely upon clicking **Sign Out** (no external database used).

---

## III. Project Structure

```text
assignment03-lumen-travel/
├── app.py                 # Main Streamlit UI, auth gating, session state & chat display
├── firebase_client.py     # Firebase Authentication (Sign Up & Sign In) REST wrapper
├── rag.py                 # Guide loader, Ollama embedding, cosine retrieval & chat generation
├── knowledge_base.json    # 20 curated entries for Zurich, Switzerland (G01–G20)
├── requirements.txt       # Python dependencies
├── .env.example           # Template for required environment variables (empty values)
├── .gitignore             # Excludes .env and local virtual environments
└── README.md              # Project documentation, setup guide, test results & limitations
```

---

## IV. Installation & Setup Instructions

### 1. Start the Ollama + Pinggy Server (Google Colab)
1. Open `Ollama_Pinggy_Server.ipynb` in Google Colab (recommended: **T4 GPU** runtime).
2. Run all cells sequentially to install Ollama, start the local server on port `11434`, and pull both `embeddinggemma` and `qwen3:4b`.
3. In the Colab terminal cell (`%xterm`), run:
   ```bash
   ssh -p 443 -R0:localhost:11434 -t qr@free.pinggy.io "u:Host:localhost:11434"
   ```
4. Copy the generated public `https://*.run.pinggy-free.link` URL.

### 2. Configure Firebase Authentication
1. Create a project in [Firebase Console](https://console.firebase.google.com/) and register a Web App (`</>`).
2. Navigate to **Authentication > Sign-in method**, enable **Email/Password**, and click **Save**.
3. Copy your Web `apiKey` and `projectId`.

### 3. Local Environment Setup
1. Clone this repository:
   ```bash
   git clone <https://github.com/BryantVinhz/Lab-Computational-Thinking>
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # macOS / Linux:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file from `.env.example` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   OLLAMA_BASE_URL=https://your-pinggy-url.run.pinggy-free.link
   FIREBASE_API_KEY=your-firebase-web-api-key
   FIREBASE_PROJECT_ID=your-firebase-project-id
   ```

### 4. Run the Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser, create an account in the **Sign Up** tab, then log in via **Sign In** to start chatting.

---

## V. Computational Thinking Overview

* **Abstraction:** Raw Zurich travel details are abstracted into structured JSON records (`id`, `title`, `text`, `source`, plus `category` and `tags` for place entries). The system maps user inputs `(email, password, question)` to grounded outputs `(answer with [Gxx] citations, top-k evidence, cosine scores)`.
* **Decomposition:** The application is decomposed into 6 sequential sub-problems with well-defined inputs and outputs:

  ```text
  [1. Sign-in] (email, password) ──> user session {email, uid, idToken}
       │
       ▼
  [2. Embed]   (20 guide texts / user question) ──> doc_vectors & query_vector 
       │
       ▼
  [3. Retrieve] (query_vector, doc_vectors, records, k=3) ──> top-3 documents + cosine scores
       │
       ▼
  [4. Build Context] (top-3 documents) ──> formatted context string "[Gxx] (Title) Text"
       │
       ▼
  [5. Generate] (question, context, SYSTEM_PROMPT) ──> grounded answer / "I don't know"
       │
       ▼
  [6. Display] (answer, documents, session_state) ──> chat bubble + expandable sources UI

* **Pattern Recognition:** 
   * **Reused Architectural Patterns (Labs 03, 04 & 05):** From **Lab 03**, the app reuses the Streamlit form/widget structure, `.env` secret management, `st.session_state` for session-gated UI access and ephemeral chat history, and the Firebase Email/Password REST authentication flow (`firebase_client.py`). From **Lab 04 & 05**, it reuses the Ollama + Pinggy HTTPS tunnel architecture, one-time document embedding via `embeddinggemma` (`/api/embed`), and vectorized cosine similarity search (`rag.py`).
   * **Core Ranking Pattern (`filter -> score -> top-k`):** Every user query follows a consistent **`filter -> score -> top-k`** ranking pattern in `retrieve()`, where the system scans the candidate pool of 20 guide entries (`G01`–`G20`), computes cosine similarity scores against the query vector, and selects the top $k=3$ passages. This exact pattern will appear again in (`RAG-Rec`) to filter places by `category` and `tags`, score candidate relevance, and return the top-$k$ recommendations. Structurally, the application also applies a uniform **session-gated state & REST request-response pattern** across external APIs (Firebase Auth and Ollama) and `st.session_state` chat turns.
* **Algorithm Design:**
  * **End-to-End Question Answering Pseudo-code:**
    ```python
    # Offline / Cached Once (@st.cache_resource)
    records <- load_guide("knowledge_base.json")          # 20 entries (G01–G20)
    doc_vectors <- embed_texts([r.title + ": " + r.text for r in records])

    # Online Query Pipeline (per user question)
    function ANSWER_QUESTION(question, records, doc_vectors, k=3):
        if trim(question) == "":
            return Warning("Please enter a non-empty question.")

        q_vec <- embed_texts([question])[0]               # query vector via embeddinggemma
        for i from 0 to len(records) - 1:
            scores[i] <- dot(doc_vectors[i], q_vec) / (norm(doc_vectors[i]) * norm(q_vec) + 1e-12)

        top_indices <- argsort_descending(scores)[0 : k]
        hits <- [ {records[i], score: scores[i]} for i in top_indices ]

        context <- join("\n\n", [ f"[{h.id}] ({h.title}) {h.text}" for h in hits ])
        prompt <- f"Evidence:\n{context}\n\nQuestion: {question}"
        answer <- stream_ollama_chat(model="qwen3:4b", system=SYSTEM_PROMPT,
                                     user=prompt, temperature=0, num_predict=650)

        display_answer_and_expander(answer, hits)
        session_state.messages.append({question, answer, hits})
    ```

  * **Key Design Decisions & Real Examples:**
   1. **Choosing Top-$k = 3$ for Retrieval:** Setting $k=3$ ensures enough context is retrieved for multi-aspect or overlapping queries without flooding the prompt with unrelated entries. For example, in **Test #6** (*"Which church has stained-glass windows by Marc Chagall, and how much is the entry fee?"*), both Fraumünster Church (`G03`) and Kunsthaus Zürich (`G08`) mention Marc Chagall; `k=3` captures `[G03, G02, G08]`, allowing the generator to disambiguate the church (`G03`, 5 CHF) from the art museum (`G08`, 24 CHF).
    2. **Closed-Book Grounding & `"I don't know"` Principle:** Because top-$k$ search always returns $k=3$ passages even when an answer is missing, `SYSTEM_PROMPT` restrict the model to rely *solely* on the supplied evidence rather than external pre-trained knowledge. For example, on an out-of-scope query like train fares to *Jungfraujoch* (**Test #7**), the assistant detects the absence of supporting facts and safely responds with `"I don't know"` instead of hallucinating.

---

## VI. Evaluation & Test Table


|  | Question | Category | Expected ID | Retrieved IDs (Top-3) | Correct? | Failure Stage (if wrong) | Notes |
| :- | :--- | :--- | :-: | :-: | :-: | :--- | :--- |
| 1 | How do I get to the Lindt Home of Chocolate from Bürkliplatz, and what are the ticket prices? | Answerable | `G10` | `[G10, G17, G04]` | `Yes` | None | Correctly states: Bus 165 (20 min), ticket prices are 17 CHF adults and 10 CHF children, advance online booking required. |
| 2 | Which trains go from Zurich Airport to Zurich Hauptbahnhof, how long is the ride, and how much does a single ticket cost? | Answerable | `G17` | `[G17, G16, G07]` | `Yes` | None | Correctly retrieves S2, S16, and InterRegio trains; ride time is 10–12 mins; the ticket cost is 7.00 CHF (Zone 110+121). |
| 3 | When is admission free at Kunsthaus Zürich, and what is the regular adult ticket price? | Answerable | `G08` | `[G08, G09, G12]` | `Yes` | None | Correctly cites free permanent collection on Wednesdays and 24 CHF adult ticket. |
| 4 | Where can I refill my bottle with free safe drinking water around Zurich? | Paraphrase | `G18` | `[G18, G20, G06]` | `Yes` | None | Matches `"refill my bottle"` semantically to `"1,200 public fountains... dispense clean, free, drinkable spring water"`. |
| 5 | Is there an outdoor spot near Hardbrücke where I can eat traditional Swiss melted cheese in winter? | Paraphrase | `G14` | `[G14, G06, G13]` | `Yes` | None | Matches `"melted cheese"` to `"traditional Swiss cheese fondue"` at Frau Gerolds Garten. |
| 6 | Which church has stained-glass windows by Marc Chagall, and how much is the entry fee? | Answerable (Edge Case) | `G03` | `[G03, G08, G02]` | `Yes` | None | Tests disambiguation between Fraumünster (`G03`, 5 CHF) and Kunsthaus (`G08`, 24 CHF), which both mention Marc Chagall. |
| 7 | How much does a round-trip train ticket from Zurich to Jungfraujoch cost? | Not in guide | `None` | `[G17, G16, G18]` | `Yes` | None | Jungfraujoch is outside the Zurich guide; assistant properly responds `"I don't know"`. |
| 8 | What are the luggage storage locker prices at Zurich Hauptbahnhof? | Not in guide | `None` | `[G17, G09, G20]` | `Yes` | None | Service not covered in `knowledge_base.json`; expected `"I don't know"`. |

---

## VII. Known Limitations

1. **No Minimum Similarity Threshold:** `retrieve()` always returns the top $k=3$ documents regardless of how low their cosine similarity scores are. For out-of-scope questions, irrelevant chunks are still passed to the LLM, relying solely on prompt compliance to say `"I don't know"`.
2. **Ephemeral Server Tunnel:** The free Pinggy HTTPS URL expires when the Google Colab session disconnects, requiring manual updates to `OLLAMA_BASE_URL` in `.env` and a Streamlit restart.
3. **Language Sensitivity:** Because `knowledge_base.json` is written in English, non-English queries yield lower cosine similarity scores and reduced retrieval accuracy.
4. **Non-Persistent Chat History:** By design, chat history lives only in `st.session_state` and is wiped upon page refresh or sign-out.

---

## VIII. AI Tools Note

* **AI Assistance:** Google Gemini was used to help create the 20 Zurich travel guide entries in `knowledge_base.json`
