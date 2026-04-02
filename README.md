# 📊 Google Ad Performance Analyzer

An **AI-powered analytics tool** that lets you query your Google Ads campaign data using **plain English**. Built with Python, Streamlit, and LLM integration (OpenAI GPT / Google Gemini).

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green?logo=openai)

---

## ✨ Features

- 🗣️ **Natural Language Queries** — Ask questions like *"Which campaigns have the highest CTR?"*
- 📊 **Auto-Generated Charts** — The LLM suggests and renders relevant visualizations
- 💡 **Proactive Insights** — Automatically surfaces actionable findings you didn't ask for
- 🔄 **Context-Aware Chat** — Follow-up questions remember the conversation
- 🛡️ **Guardrails** — Blocks prompt injection, off-topic queries, and hallucination
- 📝 **Query Logging** — Every interaction is logged for audit
- ⚡ **Cached Data Loading** — Fast responses after initial load
- 🎨 **Premium Dark UI** — Glassmorphism design with gradient accents

---

## 📁 Project Structure

```
GOOGLE AD PERFORMANCE ANALYZER/
├── app.py                  # Streamlit UI (main entry point)
├── config.py               # Centralized configuration
├── data_loader.py          # CSV loading module
├── preprocess.py           # Data cleaning & feature engineering
├── llm_engine.py           # LLM query engine (OpenAI / Gemini)
├── insight_generator.py    # Proactive insight generation
├── guardrails.py           # Input validation & safety
├── query_logger.py         # JSONL query logging
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .env                    # Your actual API keys (not committed)
├── GoogleAds_DataAnalytics_Sales_Uncleaned.csv  # Dataset
├── logs/                   # Auto-created query logs
└── README.md               # This file
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** installed
- An **API key** for either:
  - [OpenAI](https://platform.openai.com/api-keys) (recommended: GPT-4o-mini)
  - [Google Gemini](https://aistudio.google.com/apikey)

### Step 1: Clone / Navigate to the project

```bash
cd "d:\coding\GOOGLE AD PERFORMANCE ANALYZER"
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # macOS/Linux
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure your API key

```bash
# Copy the example env file
copy .env.example .env
```

Then edit `.env` and set your API key:

```ini
# For OpenAI (default)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-actual-key-here

# OR for Gemini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-actual-key-here
```

### Step 5: Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 💬 Example Questions

| Question | What it does |
|----------|-------------|
| *"Which campaigns have the highest CTR?"* | Ranks campaigns by click-through rate |
| *"What is the trend in spend over time?"* | Shows cost trends with a line chart |
| *"Which device performs best?"* | Compares desktop, mobile, tablet |
| *"Top 5 keywords by ROI"* | Finds most profitable keywords |
| *"Compare weekend vs weekday performance"* | Analyzes day-of-week patterns |
| *"Which ads have high spend but low conversions?"* | Flags inefficient ad spend |

---

## 🏗️ Architecture

```
User Question → Guardrails → LLM Engine → Structured Response
                                ↕                    ↓
                          Dataset Context      Chart Rendering
                          (schema + samples)         ↓
                                              Proactive Insights
                                                     ↓
                                              Query Logger
```

### Data Pipeline
1. **data_loader.py** — Reads the raw CSV
2. **preprocess.py** — Cleans dates, normalizes names, handles missing values, engineers features (CTR, CPC, ROI)
3. Data is cached in Streamlit for 5 minutes

### LLM Pipeline
1. **guardrails.py** — Validates input (length, topic relevance, injection detection)
2. **llm_engine.py** — Builds structured prompt with schema + sample rows, sends to LLM
3. **insight_generator.py** — Generates additional proactive insights (rule-based + LLM)
4. **query_logger.py** — Logs every interaction to `logs/query_log.jsonl`

---

## ⚙️ Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` or `gemini` |
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model to use |
| `GEMINI_API_KEY` | — | Your Google Gemini key |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model to use |
| `DATASET_PATH` | `GoogleAds_...csv` | Path to dataset file |

---

## 📊 Dataset Details

The included dataset contains **2,601 Google Ads records** with:
- **Ad_ID** — Unique identifier
- **Campaign_Name** — Campaign name (various spellings, normalized during preprocessing)
- **Clicks, Impressions** — Engagement metrics
- **Cost** — Ad spend (USD)
- **Leads, Conversions** — Conversion funnel
- **Sale_Amount** — Revenue generated
- **Ad_Date** — Date (mixed formats, unified during preprocessing)
- **Location** — Geographic targeting
- **Device** — Desktop, mobile, tablet
- **Keyword** — Search keyword

### Engineered Features
- **CTR** — Click-through rate (clicks / impressions)
- **CPC** — Cost per click (cost / clicks)
- **Cost per Conversion** — Efficiency metric
- **ROI** — Return on investment ((revenue - cost) / cost)

---

## 🔒 Safety & Guardrails

- Prompt injection patterns are blocked
- Off-topic queries are rejected with helpful suggestions
- LLM is instructed to ONLY use dataset information
- All queries are logged for audit

---

## 📄 License

This project is for educational and portfolio purposes.
