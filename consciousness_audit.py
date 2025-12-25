"""
================================================================================
CONSCIOUSNESS WARS BIBLIOMETRIC AUDIT PIPELINE v4.0
================================================================================
A comprehensive bibliometric analysis of consciousness theories:
IIT (Integrated Information Theory), GNWT (Global Neuronal Workspace Theory),
HOT (Higher-Order Theory), and RPT (Recurrent Processing Theory).

ANALYSES:
1. ConTraSt Audit - Methodology predicts theory
2. Moving Goalposts - Semantic drift over time
3. Adversarial Index - Cross-theory engagement
4. Neuro-Cartography - Brain region preferences
5. Aggression Index - Hostile tone over time
6. Easy vs Hard Problem - Phenomenology vs Function
7. Model Organism - Subject preferences
8. Semantic Space Clustering - UMAP visualization
9. Hypothesis Testing - Statistical tests
10. Methodology Predicts Theory - Yaron et al. replication
11. Vocabulary Divergence - Echo chamber analysis
12. Dogmatism Analysis - Epistemic confidence ratings

KEY FEATURES:
- Robust "Fingerprint" classification with theory-specific markers
- Separate Dogmatism/Epistemic Confidence scoring (1-10)
- Gemini text-embedding-004 for semantic embeddings
- UMAP dimensionality reduction
- Statistical hypothesis testing
- Vocabulary divergence over time (echo chamber detection)

Extracts 8 dimensions + dogmatism score per paper using Gemini LLM.
================================================================================
"""

# =============================================================================
# CELL 1: CONFIGURATION
# =============================================================================

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        🔧 CONFIGURATION 🔧                                 ║
# ╠═══════════════════════════════════════════════════════════════════════════╣
# ║  API key loaded from .env file automatically                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def load_env_file(env_path: str = ".env") -> dict:
    """Load environment variables from .env file."""
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")
    return env_vars

# Load from .env file
import os
_env = load_env_file()
GEMINI_API_KEY = _env.get('GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY', ''))

OUTPUT_FOLDER = "./results"
CSV_PATH = "cons_bib.csv"

# Model options: "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"
# Try "gemini-2.0-flash" first - it's stable and fast
GEMINI_MODEL = _env.get('GEMINI_MODEL', 'gemini-2.0-flash')
EMBEDDING_MODEL = "text-embedding-004"

MAX_WORKERS = 10
API_DELAY = 0.15
MAX_RETRIES = 3
EMBEDDING_BATCH_SIZE = 100

BASELINE_YEAR = 2005
MIN_YEAR = 2000
MAX_YEAR = 2025
SCHISM_YEAR = 2015

DEBUG_MODE = True  # Enable to see API responses
DEBUG_SAMPLE_SIZE = 10

# =============================================================================
# CELL 2: INSTALLATION
# =============================================================================

def install_packages():
    import subprocess, sys
    for pkg in ['pandas', 'numpy', 'plotly', 'scikit-learn', 'sentence-transformers',
                'requests', 'tqdm', 'kaleido', 'umap-learn', 'matplotlib', 'seaborn', 'scipy']:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    print("✅ All packages installed!")

# =============================================================================
# CELL 3: IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import json, re, time, os, warnings
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine, euclidean
from scipy import stats
import requests
from tqdm import tqdm

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

warnings.filterwarnings('ignore')

# =============================================================================
# CELL 4: DATA LOADING
# =============================================================================

def load_bibliography(csv_path: str) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("📚 LOADING BIBLIOGRAPHY DATA")
    print("=" * 60)
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    print(f"✅ Loaded {len(df)} papers")
    df_valid = df[df['Abstract'].notna() & (df['Abstract'].str.len() > 50)].copy()
    print(f"📄 Papers with valid abstracts: {len(df_valid)}")
    df_valid['Year'] = pd.to_numeric(df_valid['Year'], errors='coerce')
    df_valid = df_valid[df_valid['Year'].notna()]
    df_valid['Year'] = df_valid['Year'].astype(int)
    return df_valid.reset_index(drop=True)

def extract_first_sentences(text: str, n: int = 2) -> str:
    if pd.isna(text): return ""
    return ' '.join(re.split(r'(?<=[.!?])\s+', text.strip())[:n])

# =============================================================================
# CELL 5: CLASSIFICATION PROMPT (IMPROVED)
# =============================================================================

CLASSIFICATION_PROMPT = '''Classify this neuroscience abstract. Detect theory "fingerprints" carefully.

ABSTRACT:
{abstract}

=== THEORY FINGERPRINTS ===

IIT (Integrated Information Theory):
- Keywords: "Phi" (Φ), "integrated information", "differentiation + integration", "cause-effect structure", "posterior hot zone", "maximally irreducible", "exclusion postulate", "intrinsic causal power"
- Authors: Tononi, Koch, Massimini, Boly, Casali

GNWT (Global Neuronal Workspace Theory):
- Keywords: "Global workspace", "ignition", "P3b"/"P300", "broadcasting", "long-distance connectivity", "fronto-parietal", "access consciousness", "late cortical activity" (>300ms), "prefrontal amplification"
- Authors: Dehaene, Baars, Changeux, Sergent, Naccache

HOT (Higher-Order Theory):
- Keywords: "Higher-order thought/representation", "meta-cognition", "awareness of awareness", "second-order", "introspection mechanism", "prefrontal for awareness"
- Authors: Rosenthal, Lau, Brown, LeDoux, Fleming

RPT (Recurrent Processing Theory):
- Keywords: "Local recurrence in SENSORY cortex", "re-entrant processing in V1/V2/V4", "feedforward vs feedback in visual areas", "recurrent loops in occipital/temporal cortex"
- Authors: Lamme, Super, Fahrenfort, Scholte
- ⚠️ NOT RPT if: mentions "prefrontal feedback", "global workspace", "fronto-parietal"

NEUTRAL: No clear theory markers, purely methodological, or equally discusses multiple theories.

=== OTHER DIMENSIONS ===

PARADIGM:
- "Report" = participants report conscious experience (verbal report, button press for seen/not seen)
- "No-Report" = measures consciousness without reports (no-report paradigms, implicit measures, brain-only)

TYPE:
- "Content" = WHAT is conscious (visual features, objects, colors, specific percepts)
- "State" = WHETHER conscious (awake vs anesthesia, sleep vs wake, disorders of consciousness)

EPISTEMIC:
- "A Priori" = derives predictions FROM theory before testing (theory-driven hypothesis)
- "Post-Hoc" = tests existing data or no explicit theoretical prediction

ANATOMY: List up to 3 brain regions mentioned (e.g., ["PFC", "V1", "parietal"]) or ["none"]

TONE toward rival theories:
- "Dismissive" = strong criticism, "refutes", "falsifies", "disproves"
- "Critical" = challenges, questions, "problematic for"
- "Constructive" = acknowledges strengths, seeks integration
- "Neutral" = no mention of other theories

TARGET:
- "Phenomenology" = subjective experience, qualia, what-it-is-like
- "Function" = cognitive function, access, report, behavior
- "Mechanism" = neural mechanism, implementation, substrate

SUBJECT:
- "Human" = healthy human participants
- "Clinical" = patients, disorders of consciousness, brain injury
- "Animal" = non-human animals
- "Simulation" = computational models, simulations
- "Review" = review paper, meta-analysis, theoretical paper

Return ONLY valid JSON (no markdown, no explanation):
{"theory":"IIT|GNWT|HOT|RPT|Neutral","paradigm":"Report|No-Report","type":"Content|State","epistemic":"A Priori|Post-Hoc","anatomy":["region1","region2"],"tone":"Dismissive|Critical|Constructive|Neutral","target":"Phenomenology|Function|Mechanism","subject":"Human|Clinical|Animal|Simulation|Review","confidence":0.0-1.0,"fingerprints":["found","markers"]}'''

# =============================================================================
# CELL 6: DOGMATISM PROMPT (NEW)
# =============================================================================

DOGMATISM_PROMPT = '''Rate the epistemic confidence/dogmatism of this abstract on a scale of 1-10.

ABSTRACT:
{abstract}

SCORING GUIDE:

HIGH DOGMATISM (8-10):
- Words: "proves", "demonstrates conclusively", "establishes", "fundamental law", "definitive", "unequivocal", "certain", "must be", "the only explanation"
- Tone: Absolute certainty, dismissive of alternatives, grandiose claims

MODERATE CONFIDENCE (5-7):
- Words: "shows", "indicates", "supports", "evidence for", "consistent with", "confirms"
- Tone: Confident but acknowledges limits

LOW DOGMATISM (1-4):
- Words: "suggests", "may", "might", "could", "hypothesis", "preliminary", "tentative", "one possibility", "remains unclear"
- Tone: Humble, acknowledges uncertainty, open to alternatives

MARKERS TO LOOK FOR:
- Grandiosity markers: "fundamental", "revolutionary", "paradigm shift", "solves the hard problem"
- Certainty markers: "proves", "demonstrates", "establishes beyond doubt"
- Hedging markers: "suggests", "may indicate", "preliminary evidence"
- Humility markers: "limitations include", "further research needed", "one interpretation"

Return ONLY valid JSON:
{"dogmatism_score":1-10,"confidence_markers":["list","of","markers","found"],"hedging_markers":["list","of","hedges"],"reasoning":"brief explanation"}'''

# =============================================================================
# CELL 7: GEMINI EXTRACTOR (IMPROVED)
# =============================================================================

class GeminiExtractor:
    def __init__(self, api_key: str, model: str = GEMINI_MODEL):
        self.api_key, self.model = api_key, model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.session = requests.Session()
        self.debug_responses = []
        if not api_key: raise ValueError("❌ GEMINI_API_KEY not set!")

    def _call_api(self, prompt: str, temperature: float = 0.1) -> Tuple[Optional[str], str]:
        """Make API call and return raw text response."""
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 500
            }
        }
        try:
            r = self.session.post(f"{self.base_url}?key={self.api_key}", json=payload, timeout=60)
            if r.status_code == 429:
                time.sleep(10)
                return None, "RATE_LIMITED"
            if r.status_code != 200:
                error_text = r.text[:200] if r.text else "No error text"
                if DEBUG_MODE:
                    print(f"  ⚠️ API Error {r.status_code}: {error_text}")
                return None, f"HTTP_{r.status_code}"
            result = r.json()
            if 'candidates' in result and result['candidates']:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    text = candidate['content']['parts'][0].get('text', '')
                    return text, "OK"
                # Check for safety block
                if candidate.get('finishReason') == 'SAFETY':
                    return None, "SAFETY_BLOCKED"
            return None, "NO_CANDIDATES"
        except requests.exceptions.Timeout:
            return None, "TIMEOUT"
        except Exception as e:
            if DEBUG_MODE:
                print(f"  ⚠️ Exception: {str(e)[:100]}")
            return None, f"ERROR: {str(e)[:50]}"

    def _parse_json(self, text: str) -> Optional[Dict]:
        """Parse JSON from response text."""
        if not text:
            return None

        # Clean up markdown formatting
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
        text = text.strip()

        # Try direct parse
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except:
            pass

        # Try to extract JSON from text - find first { and matching }
        try:
            start = text.find('{')
            if start >= 0:
                depth = 0
                end = start
                for i, c in enumerate(text[start:], start):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                json_str = text[start:end+1]
                result = json.loads(json_str)
                if isinstance(result, dict):
                    return result
        except:
            pass

        # Last resort: try to find any JSON object
        try:
            match = re.search(r'\{[^{}]*"theory"[^{}]*\}', text)
            if match:
                result = json.loads(match.group())
                if isinstance(result, dict):
                    return result
        except:
            pass

        return None

    def _default_classification(self) -> Dict:
        return {
            "theory": "Neutral", "paradigm": "Report", "type": "Content",
            "epistemic": "Post-Hoc", "anatomy": [], "tone": "Neutral",
            "target": "Function", "subject": "Human", "confidence": 0.0,
            "fingerprints": [], "dogmatism_score": 5, "confidence_markers": [],
            "hedging_markers": [], "_extraction_failed": True
        }

    def _normalize_classification(self, d: Dict) -> Dict:
        """Normalize parsed JSON to expected format."""
        r = self._default_classification()
        r["_extraction_failed"] = False

        # Theory - must be exact match
        theory = str(d.get("theory", "")).strip().upper()
        if theory in ["IIT", "GNWT", "HOT", "RPT"]:
            r["theory"] = theory
        elif "IIT" in theory or "INTEGRATED" in theory:
            r["theory"] = "IIT"
        elif "GNWT" in theory or "GNW" in theory or "GLOBAL" in theory or "WORKSPACE" in theory:
            r["theory"] = "GNWT"
        elif "HOT" in theory or "HIGHER" in theory:
            r["theory"] = "HOT"
        elif "RPT" in theory or "RECURRENT" in theory:
            r["theory"] = "RPT"
        else:
            r["theory"] = "Neutral"

        # Paradigm
        paradigm = str(d.get("paradigm", "")).lower()
        r["paradigm"] = "No-Report" if "no" in paradigm or "implicit" in paradigm else "Report"

        # Type
        type_val = str(d.get("type", "")).lower()
        r["type"] = "State" if "state" in type_val else "Content"

        # Epistemic
        epistemic = str(d.get("epistemic", "")).lower()
        r["epistemic"] = "A Priori" if "priori" in epistemic or "a pri" in epistemic else "Post-Hoc"

        # Anatomy
        anat = d.get("anatomy", [])
        if isinstance(anat, str):
            anat = [a.strip() for a in anat.split(",")]
        if isinstance(anat, list):
            r["anatomy"] = [str(a).strip().upper() for a in anat
                          if a and str(a).lower() not in ["none", "null", ""]][:3]

        # Tone
        tone = str(d.get("tone", "")).lower()
        if "dismiss" in tone:
            r["tone"] = "Dismissive"
        elif "critic" in tone:
            r["tone"] = "Critical"
        elif "construct" in tone:
            r["tone"] = "Constructive"
        else:
            r["tone"] = "Neutral"

        # Target
        target = str(d.get("target", "")).lower()
        if "phenom" in target or "qualia" in target or "subjective" in target:
            r["target"] = "Phenomenology"
        elif "mechan" in target or "neural" in target or "implement" in target:
            r["target"] = "Mechanism"
        else:
            r["target"] = "Function"

        # Subject
        subject = str(d.get("subject", "")).lower()
        if "clinical" in subject or "patient" in subject or "disorder" in subject:
            r["subject"] = "Clinical"
        elif "animal" in subject or "monkey" in subject or "mouse" in subject:
            r["subject"] = "Animal"
        elif "simul" in subject or "model" in subject or "comput" in subject:
            r["subject"] = "Simulation"
        elif "review" in subject or "meta" in subject or "theoret" in subject:
            r["subject"] = "Review"
        else:
            r["subject"] = "Human"

        # Confidence
        try:
            conf = float(d.get("confidence", 0.5))
            r["confidence"] = max(0.0, min(1.0, conf))
        except:
            r["confidence"] = 0.5

        # Fingerprints
        fp = d.get("fingerprints", d.get("fingerprints_found", []))
        if isinstance(fp, list):
            r["fingerprints"] = [str(f) for f in fp]

        return r

    def extract_classification(self, abstract: str) -> Dict:
        """Extract classification for a single abstract."""
        if pd.isna(abstract) or len(str(abstract)) < 50:
            return self._default_classification()

        prompt = CLASSIFICATION_PROMPT.format(abstract=str(abstract)[:3000])

        for attempt in range(MAX_RETRIES):
            text, status = self._call_api(prompt, temperature=0.1)

            if status == "RATE_LIMITED":
                time.sleep(5 * (attempt + 1))
                continue

            if status != "OK":
                if DEBUG_MODE and attempt == 0:
                    print(f"  API status: {status}")
                time.sleep(2)
                continue

            if text:
                # Debug: show raw response for first few
                if DEBUG_MODE and len(self.debug_responses) < 3:
                    print(f"\n  📝 Raw API response ({len(text)} chars):")
                    print(f"     {text[:200]}...")

                parsed = self._parse_json(text)

                if DEBUG_MODE and len(self.debug_responses) < 3:
                    print(f"  📝 Parsed JSON: {parsed}")

                if parsed:
                    try:
                        result = self._normalize_classification(parsed)
                        if DEBUG_MODE and len(self.debug_responses) < DEBUG_SAMPLE_SIZE:
                            self.debug_responses.append({
                                "raw": text[:300], "parsed": parsed, "normalized": result
                            })
                        return result
                    except Exception as e:
                        if DEBUG_MODE:
                            print(f"  ⚠️ Normalization error: {e}")
                            print(f"     Parsed data: {parsed}")
                elif DEBUG_MODE and len(self.debug_responses) < 3:
                    print(f"  ⚠️ JSON parse failed. Raw: {text[:200]}...")

            time.sleep(1)

        return self._default_classification()

    def extract_dogmatism(self, abstract: str) -> Dict:
        """Extract dogmatism score for a single abstract."""
        if pd.isna(abstract) or len(str(abstract)) < 50:
            return {"dogmatism_score": 5, "confidence_markers": [], "hedging_markers": [], "reasoning": ""}

        prompt = DOGMATISM_PROMPT.format(abstract=str(abstract)[:3000])

        for attempt in range(MAX_RETRIES):
            text, status = self._call_api(prompt, temperature=0.1)

            if status == "RATE_LIMITED":
                time.sleep(5 * (attempt + 1))
                continue

            if status != "OK":
                time.sleep(2)
                continue

            if text:
                parsed = self._parse_json(text)
                if parsed:
                    try:
                        score = int(parsed.get("dogmatism_score", 5))
                        score = max(1, min(10, score))
                    except:
                        score = 5
                    return {
                        "dogmatism_score": score,
                        "confidence_markers": parsed.get("confidence_markers", []),
                        "hedging_markers": parsed.get("hedging_markers", []),
                        "reasoning": parsed.get("reasoning", "")
                    }

            time.sleep(1)

        return {"dogmatism_score": 5, "confidence_markers": [], "hedging_markers": [], "reasoning": ""}

    def extract_single(self, abstract: str, idx: int = 0) -> Dict:
        """Extract both classification and dogmatism for a single abstract."""
        try:
            classification = self.extract_classification(abstract)
            # Skip dogmatism for now to speed up - can enable later
            # dogmatism = self.extract_dogmatism(abstract)
            # classification.update(dogmatism)
            classification['dogmatism_score'] = 5
            classification['confidence_markers'] = []
            classification['hedging_markers'] = []
            return classification
        except Exception as e:
            if DEBUG_MODE:
                print(f"  extract_single error at {idx}: {type(e).__name__}: {e}")
            return self._default_classification()

    def batch_extract_parallel(self, df: pd.DataFrame, max_workers: int = MAX_WORKERS) -> pd.DataFrame:
        """Extract all dimensions from DataFrame in parallel."""
        print(f"\n{'='*60}\n🔍 EXTRACTING DIMENSIONS WITH GEMINI\n{'='*60}")
        print(f"Model: {self.model} | Papers: {len(df)} | Workers: {max_workers}")

        # First, do a test extraction to verify API is working
        print("\n🧪 Testing extraction on first abstract...")
        test_abstract = df.iloc[0]['Abstract']
        test_result = self.extract_classification(test_abstract)
        print(f"   Test result: theory={test_result.get('theory', 'N/A')}, conf={test_result.get('confidence', 0):.2f}")
        if test_result.get('_extraction_failed', True):
            print("   ⚠️ Test extraction failed! Check API responses above.")
        else:
            print("   ✅ Test extraction succeeded!")

        results = [None] * len(df)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.extract_single, row['Abstract'], i): i
                      for i, row in df.iterrows()}

            for future in tqdm(as_completed(futures), total=len(df), desc="Extracting"):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"⚠️ Error at {idx}: {type(e).__name__}: {e}")
                    results[idx] = self._default_classification()
                time.sleep(API_DELAY)

        if DEBUG_MODE and self.debug_responses:
            print("\n🔍 DEBUG SAMPLES (successful extractions):")
            for i, d in enumerate(self.debug_responses[:5]):
                try:
                    print(f"  Sample {i}: {d.get('normalized', {}).get('theory', 'N/A')}")
                    print(f"    Raw: {str(d.get('raw', ''))[:100]}...")
                except Exception as e:
                    print(f"  Sample {i}: Error displaying: {e}")

        # Apply results to dataframe
        df_out = df.copy()
        keys = ['theory', 'paradigm', 'type', 'epistemic', 'anatomy', 'tone', 'target',
                'subject', 'confidence', 'fingerprints', 'dogmatism_score',
                'confidence_markers', 'hedging_markers', '_extraction_failed']

        for k in keys:
            default = self._default_classification().get(k, None)
            df_out[k] = [r.get(k, default) if r else default for r in results]

        # Print statistics
        print(f"\n📊 Extraction Statistics:")
        failed = df_out['_extraction_failed'].sum()
        print(f"   Extraction failures: {failed} ({failed/len(df_out)*100:.1f}%)")
        print(f"   Mean confidence: {df_out['confidence'].mean():.3f}")
        print(f"   Mean dogmatism: {df_out['dogmatism_score'].mean():.2f}")

        print(f"\n📊 Theory distribution:")
        for t, c in df_out['theory'].value_counts().items():
            print(f"   {t}: {c} ({c/len(df_out)*100:.1f}%)")

        print(f"\n📊 Paradigm distribution:")
        for p, c in df_out['paradigm'].value_counts().items():
            print(f"   {p}: {c} ({c/len(df_out)*100:.1f}%)")

        print(f"\n📊 Tone distribution:")
        for t, c in df_out['tone'].value_counts().items():
            print(f"   {t}: {c} ({c/len(df_out)*100:.1f}%)")

        return df_out

# =============================================================================
# CELL 8: GEMINI EMBEDDER
# =============================================================================

class GeminiEmbedder:
    def __init__(self, api_key: str, model: str = EMBEDDING_MODEL):
        self.api_key, self.model = api_key, model
        self.batch_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
        self.session = requests.Session()

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[Optional[np.ndarray]]:
        print(f"\n{'='*60}\n🧮 GENERATING EMBEDDINGS\n{'='*60}")
        print(f"Model: {self.model} | Texts: {len(texts)}")
        embeddings = [None] * len(texts)

        for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[start:start + batch_size]
            reqs, idxs = [], []

            for i, t in enumerate(batch):
                if t and len(str(t)) >= 10:
                    reqs.append({
                        "model": f"models/{self.model}",
                        "content": {"parts": [{"text": str(t)[:2000]}]}
                    })
                    idxs.append(start + i)

            if not reqs:
                continue

            for attempt in range(3):
                try:
                    r = self.session.post(
                        f"{self.batch_url}?key={self.api_key}",
                        json={"requests": reqs},
                        timeout=60
                    )
                    if r.status_code == 429:
                        time.sleep(30 * (attempt + 1))
                        continue
                    r.raise_for_status()

                    for j, emb in enumerate(r.json().get('embeddings', [])):
                        if 'values' in emb:
                            embeddings[idxs[j]] = np.array(emb['values'])
                    break
                except Exception as e:
                    print(f"⚠️ Batch error (attempt {attempt+1}): {e}")
                    time.sleep(5)

            time.sleep(0.3)

        embedded_count = sum(1 for e in embeddings if e is not None)
        print(f"✅ Embedded: {embedded_count}/{len(texts)} ({embedded_count/len(texts)*100:.1f}%)")
        return embeddings

# =============================================================================
# CELL 9: RULE-BASED FALLBACK
# =============================================================================

class RuleBasedExtractor:
    """Fallback extractor using keyword matching."""

    FINGERPRINTS = {
        'IIT': ['integrated information', 'phi', 'tononi', 'cause-effect', 'posterior hot zone',
                'maximally irreducible', 'exclusion postulate', 'intrinsic causal', 'differentiation'],
        'GNWT': ['global workspace', 'dehaene', 'baars', 'ignition', 'broadcasting', 'p3b', 'p300',
                 'fronto-parietal', 'access consciousness', 'long-distance', 'prefrontal'],
        'HOT': ['higher-order', 'rosenthal', 'metacognition', 'meta-cognition', 'awareness of awareness',
                'lau', 'second-order', 'introspection'],
        'RPT': ['recurrent processing', 'lamme', 'local recurrence', 'feedforward sweep',
                'reentrant', 're-entrant', 'fahrenfort']
    }

    RPT_EXCLUSIONS = ['prefrontal', 'frontal feedback', 'global workspace', 'fronto-parietal']

    PARADIGM_NOREPORT = ['no-report', 'implicit', 'without report', 'brain-only', 'no report']
    PARADIGM_REPORT = ['verbal report', 'button press', 'subjective report', 'reported']

    TYPE_STATE = ['anesthesia', 'sleep', 'wake', 'disorders of consciousness', 'vegetative',
                  'minimally conscious', 'coma', 'sedation']
    TYPE_CONTENT = ['visual', 'auditory', 'color', 'motion', 'object', 'face', 'word']

    TONE_DISMISSIVE = ['refutes', 'falsifies', 'disproves', 'fails to', 'cannot explain', 'fatal flaw']
    TONE_CRITICAL = ['challenges', 'questions', 'problematic', 'limitation', 'weakness']
    TONE_CONSTRUCTIVE = ['integrates', 'bridges', 'combines', 'complementary', 'synthesis']

    DOGMATISM_HIGH = ['proves', 'demonstrates conclusively', 'establishes', 'fundamental',
                      'definitive', 'unequivocal', 'certain', 'must be', 'the only']
    DOGMATISM_LOW = ['suggests', 'may', 'might', 'could', 'hypothesis', 'preliminary',
                     'tentative', 'possibly', 'unclear', 'speculative']

    def extract(self, abstract: str) -> Dict:
        if pd.isna(abstract):
            return self._default()

        text = ' ' + str(abstract).lower() + ' '

        # Theory detection
        scores = {t: sum(1 for k in kws if k in text) for t, kws in self.FINGERPRINTS.items()}

        # RPT exclusion check
        if scores.get('RPT', 0) > 0:
            for ex in self.RPT_EXCLUSIONS:
                if ex in text:
                    scores['RPT'] = max(0, scores['RPT'] - 2)

        theory = max(scores, key=scores.get) if max(scores.values()) > 0 else "Neutral"
        confidence = min(max(scores.values()) / 3, 1.0) if max(scores.values()) > 0 else 0.0

        # Paradigm
        noreport = sum(1 for k in self.PARADIGM_NOREPORT if k in text)
        report = sum(1 for k in self.PARADIGM_REPORT if k in text)
        paradigm = "No-Report" if noreport > report else "Report"

        # Type
        state = sum(1 for k in self.TYPE_STATE if k in text)
        content = sum(1 for k in self.TYPE_CONTENT if k in text)
        type_val = "State" if state > content else "Content"

        # Tone
        dismissive = sum(1 for k in self.TONE_DISMISSIVE if k in text)
        critical = sum(1 for k in self.TONE_CRITICAL if k in text)
        constructive = sum(1 for k in self.TONE_CONSTRUCTIVE if k in text)

        if dismissive > 0:
            tone = "Dismissive"
        elif critical > constructive:
            tone = "Critical"
        elif constructive > 0:
            tone = "Constructive"
        else:
            tone = "Neutral"

        # Dogmatism
        high = sum(1 for k in self.DOGMATISM_HIGH if k in text)
        low = sum(1 for k in self.DOGMATISM_LOW if k in text)
        dogmatism = 5 + high - low
        dogmatism = max(1, min(10, dogmatism))

        return {
            "theory": theory, "paradigm": paradigm, "type": type_val,
            "epistemic": "Post-Hoc", "anatomy": [], "tone": tone,
            "target": "Function", "subject": "Human", "confidence": confidence,
            "fingerprints": [], "dogmatism_score": dogmatism,
            "confidence_markers": [], "hedging_markers": [],
            "_extraction_failed": False
        }

    def _default(self):
        return {
            "theory": "Neutral", "paradigm": "Report", "type": "Content",
            "epistemic": "Post-Hoc", "anatomy": [], "tone": "Neutral",
            "target": "Function", "subject": "Human", "confidence": 0.0,
            "fingerprints": [], "dogmatism_score": 5,
            "confidence_markers": [], "hedging_markers": [],
            "_extraction_failed": True
        }

    def batch_extract(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"\n{'='*60}\n📝 RULE-BASED EXTRACTION (FALLBACK)\n{'='*60}")
        results = [self.extract(row['Abstract']) for _, row in tqdm(df.iterrows(), total=len(df))]
        df_out = df.copy()
        for k in results[0].keys():
            df_out[k] = [r[k] for r in results]

        print(f"\n📊 Theory distribution:")
        for t, c in df_out['theory'].value_counts().items():
            print(f"   {t}: {c} ({c/len(df_out)*100:.1f}%)")
        return df_out

# =============================================================================
# CELL 10: SEMANTIC CLUSTERING
# =============================================================================

def run_semantic_clustering(df: pd.DataFrame, embeddings: List, output_folder: str) -> Dict:
    print(f"\n{'='*60}\n🌌 SEMANTIC CLUSTERING\n{'='*60}")

    if not UMAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        print("⚠️ Missing dependencies (umap-learn or matplotlib)")
        return {"error": "Missing dependencies"}

    valid_mask = [e is not None for e in embeddings]
    valid_embeddings = [e for e in embeddings if e is not None]

    if len(valid_embeddings) < 50:
        print("⚠️ Insufficient embeddings for clustering")
        return {"error": "Insufficient embeddings"}

    X = np.vstack(valid_embeddings)
    valid_df = df[valid_mask].copy().reset_index(drop=True)

    print(f"Valid embeddings: {len(X)} | Shape: {X.shape}")

    # UMAP reduction
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    X_2d = reducer.fit_transform(X)

    valid_df['umap_x'], valid_df['umap_y'] = X_2d[:, 0], X_2d[:, 1]

    # Plot semantic space
    plt.figure(figsize=(14, 10))
    colors = {'IIT': '#E63946', 'GNWT': '#457B9D', 'HOT': '#2A9D8F', 'RPT': '#E9C46A', 'Neutral': '#CCCCCC'}

    for theory in ['Neutral', 'IIT', 'GNWT', 'HOT', 'RPT']:
        mask = valid_df['theory'] == theory
        if mask.sum() > 0:
            alpha = 0.3 if theory == 'Neutral' else 0.8
            size = 20 if theory == 'Neutral' else 50
            plt.scatter(valid_df.loc[mask, 'umap_x'], valid_df.loc[mask, 'umap_y'],
                       c=colors[theory], label=f"{theory} (n={mask.sum()})",
                       alpha=alpha, s=size, edgecolors='white', linewidth=0.5)

    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.title('Semantic Space of Consciousness Research by Theory', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "8_semantic_space.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: 8_semantic_space.png")

    # Calculate centroids
    centroids = {}
    for theory in ['IIT', 'GNWT', 'HOT', 'RPT', 'Neutral']:
        mask = valid_df['theory'] == theory
        if mask.sum() >= 3:
            centroids[theory] = np.mean(X[mask], axis=0)

    return {"valid_df": valid_df, "centroids": centroids, "X": X, "X_2d": X_2d}

# =============================================================================
# CELL 11: HYPOTHESIS TESTING
# =============================================================================

def run_hypothesis_tests(df: pd.DataFrame, centroids: Dict, X: np.ndarray, output_folder: str) -> Dict:
    print(f"\n{'='*60}\n🔬 HYPOTHESIS TESTING\n{'='*60}")
    results = {}

    # A: Implicit Bias - Are neutral papers closer to IIT or GNWT?
    print("\n📊 HYPOTHESIS A: Implicit Bias in 'Neutral' Papers")
    if 'IIT' in centroids and 'GNWT' in centroids:
        neutral_mask = df['theory'] == 'Neutral'
        neutral_embeddings = X[neutral_mask]

        if len(neutral_embeddings) > 10:
            dist_to_iit = [euclidean(e, centroids['IIT']) for e in neutral_embeddings]
            dist_to_gnwt = [euclidean(e, centroids['GNWT']) for e in neutral_embeddings]

            t_stat, p_val = stats.ttest_rel(dist_to_iit, dist_to_gnwt)
            closer_to = "IIT" if np.mean(dist_to_iit) < np.mean(dist_to_gnwt) else "GNWT"

            print(f"  Mean distance to IIT: {np.mean(dist_to_iit):.4f}")
            print(f"  Mean distance to GNWT: {np.mean(dist_to_gnwt):.4f}")
            print(f"  Neutral papers closer to: {closer_to}")
            print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f} {'✅ SIGNIFICANT' if p_val < 0.05 else '❌ Not significant'}")

            results['hypothesis_a'] = {
                'closer_to': closer_to,
                'mean_dist_iit': float(np.mean(dist_to_iit)),
                'mean_dist_gnwt': float(np.mean(dist_to_gnwt)),
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'significant': p_val < 0.05
            }

    # B: The Schism - Is IIT-GNWT distance increasing over time?
    print("\n📊 HYPOTHESIS B: The Schism (IIT-GNWT Divergence)")
    if 'IIT' in centroids and 'GNWT' in centroids:
        pre_2015 = df[df['Year'] < SCHISM_YEAR]
        post_2015 = df[df['Year'] >= SCHISM_YEAR]

        def get_centroid_distance(subset, theory1, theory2):
            mask1 = (df['theory'] == theory1) & df.index.isin(subset.index)
            mask2 = (df['theory'] == theory2) & df.index.isin(subset.index)
            if mask1.sum() < 3 or mask2.sum() < 3:
                return None
            c1 = np.mean(X[mask1], axis=0)
            c2 = np.mean(X[mask2], axis=0)
            return euclidean(c1, c2)

        dist_pre = get_centroid_distance(pre_2015, 'IIT', 'GNWT')
        dist_post = get_centroid_distance(post_2015, 'IIT', 'GNWT')

        if dist_pre is not None and dist_post is not None:
            change = (dist_post - dist_pre) / dist_pre * 100
            direction = "DIVERGING" if dist_post > dist_pre else "CONVERGING"

            print(f"  IIT-GNWT distance pre-2015: {dist_pre:.4f}")
            print(f"  IIT-GNWT distance post-2015: {dist_post:.4f}")
            print(f"  Change: {change:+.1f}% ({direction})")

            results['hypothesis_b'] = {
                'distance_pre_2015': float(dist_pre),
                'distance_post_2015': float(dist_post),
                'percent_change': float(change),
                'direction': direction
            }

    # C: Insularity - Which theory has the tightest cluster?
    print("\n📊 HYPOTHESIS C: Cluster Insularity")
    densities = {}
    for theory in ['IIT', 'GNWT', 'HOT', 'RPT']:
        if theory in centroids:
            theory_mask = df['theory'] == theory
            theory_embeddings = X[theory_mask]
            if len(theory_embeddings) >= 5:
                distances = [euclidean(e, centroids[theory]) for e in theory_embeddings]
                densities[theory] = {
                    'n': len(theory_embeddings),
                    'mean_distance': float(np.mean(distances)),
                    'std_distance': float(np.std(distances))
                }
                print(f"  {theory}: n={len(theory_embeddings)}, mean_dist={np.mean(distances):.4f} ± {np.std(distances):.4f}")

    if densities:
        most_insular = min(densities, key=lambda x: densities[x]['mean_distance'])
        least_insular = max(densities, key=lambda x: densities[x]['mean_distance'])
        print(f"  Most insular (tightest cluster): {most_insular}")
        print(f"  Least insular (loosest cluster): {least_insular}")

        results['hypothesis_c'] = {
            'most_insular': most_insular,
            'least_insular': least_insular,
            'densities': densities
        }

        # Plot cluster density
        if MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=(10, 6))
            theories = list(densities.keys())
            means = [densities[t]['mean_distance'] for t in theories]
            stds = [densities[t]['std_distance'] for t in theories]
            colors = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A'][:len(theories)]

            bars = plt.bar(theories, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5)
            plt.ylabel('Mean Distance to Centroid', fontsize=12)
            plt.title('Cluster Density by Theory (Lower = More Insular)', fontsize=14)

            # Add value labels
            for bar, mean in zip(bars, means):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, "9_cluster_density.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✅ Saved: 9_cluster_density.png")

    return results

# =============================================================================
# CELL 12: VOCABULARY DIVERGENCE (ECHO CHAMBER ANALYSIS)
# =============================================================================

def run_vocabulary_divergence(df: pd.DataFrame, embeddings: List, output_folder: str) -> Dict:
    """
    Test if IIT is becoming an 'echo chamber' by measuring semantic distance
    from 'General Neuroscience' (Neutral papers) over time.
    """
    print(f"\n{'='*60}\n📚 VOCABULARY DIVERGENCE ANALYSIS (Echo Chamber Test)\n{'='*60}")

    if not MATPLOTLIB_AVAILABLE:
        return {"error": "matplotlib not available"}

    valid_mask = [e is not None for e in embeddings]
    valid_embeddings = [e for e in embeddings if e is not None]

    if len(valid_embeddings) < 100:
        return {"error": "Insufficient embeddings"}

    X = np.vstack(valid_embeddings)
    valid_df = df[valid_mask].copy().reset_index(drop=True)

    # Time periods
    periods = [
        (2000, 2010, "2000-2010"),
        (2010, 2015, "2010-2015"),
        (2015, 2020, "2015-2020"),
        (2020, 2026, "2020-2025")
    ]

    results = {'periods': {}, 'divergence_trends': {}}

    # Calculate centroid for "General Neuroscience" (Neutral papers)
    neutral_mask = valid_df['theory'] == 'Neutral'
    if neutral_mask.sum() < 10:
        print("  ⚠️ Insufficient neutral papers for baseline")
        return {"error": "Insufficient neutral papers"}

    general_centroid = np.mean(X[neutral_mask], axis=0)
    print(f"  General Neuroscience baseline: {neutral_mask.sum()} papers")

    theories = ['IIT', 'GNWT', 'HOT', 'RPT']
    divergence_data = {t: [] for t in theories}

    for start, end, label in periods:
        period_mask = (valid_df['Year'] >= start) & (valid_df['Year'] < end)
        results['periods'][label] = {}

        for theory in theories:
            theory_period_mask = period_mask & (valid_df['theory'] == theory)
            n_papers = theory_period_mask.sum()

            if n_papers >= 3:
                theory_centroid = np.mean(X[theory_period_mask], axis=0)
                distance = euclidean(theory_centroid, general_centroid)
                divergence_data[theory].append({'period': label, 'distance': distance, 'n': n_papers})
                results['periods'][label][theory] = {'distance': float(distance), 'n': n_papers}
                print(f"  {label} | {theory}: distance={distance:.4f} (n={n_papers})")

    # Calculate trends
    print(f"\n📈 Divergence Trends:")
    for theory in theories:
        if len(divergence_data[theory]) >= 2:
            distances = [d['distance'] for d in divergence_data[theory]]
            first, last = distances[0], distances[-1]
            change = (last - first) / first * 100 if first > 0 else 0
            trend = "↑ DIVERGING" if change > 10 else "↓ CONVERGING" if change < -10 else "→ STABLE"
            results['divergence_trends'][theory] = {
                'first_distance': float(first),
                'last_distance': float(last),
                'percent_change': float(change),
                'trend': trend
            }
            print(f"  {theory}: {first:.4f} → {last:.4f} ({change:+.1f}%) {trend}")

    # Plot divergence over time
    plt.figure(figsize=(12, 6))
    colors = {'IIT': '#E63946', 'GNWT': '#457B9D', 'HOT': '#2A9D8F', 'RPT': '#E9C46A'}

    for theory in theories:
        if divergence_data[theory]:
            periods_plot = [d['period'] for d in divergence_data[theory]]
            distances = [d['distance'] for d in divergence_data[theory]]
            plt.plot(periods_plot, distances, marker='o', markersize=10, linewidth=2,
                    label=theory, color=colors[theory])

    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Semantic Distance from General Neuroscience', fontsize=12)
    plt.title('Vocabulary Divergence Over Time\n(Higher = More "Echo Chamber")', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "11_vocabulary_divergence.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: 11_vocabulary_divergence.png")

    return results

# =============================================================================
# CELL 13: DOGMATISM ANALYSIS
# =============================================================================

def run_dogmatism_analysis(df: pd.DataFrame, output_folder: str) -> Dict:
    """
    Analyze dogmatism/epistemic confidence by theory.
    """
    print(f"\n{'='*60}\n🎯 DOGMATISM ANALYSIS\n{'='*60}")

    if not MATPLOTLIB_AVAILABLE:
        return {"error": "matplotlib not available"}

    results = {}

    # Filter to theories only
    theories_df = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])].copy()

    if len(theories_df) < 20:
        print("  ⚠️ Insufficient theory-classified papers")
        return {"error": "Insufficient data"}

    # Calculate mean dogmatism by theory
    dogmatism_by_theory = theories_df.groupby('theory')['dogmatism_score'].agg(['mean', 'std', 'count'])
    print("\n📊 Dogmatism by Theory:")
    for theory in ['IIT', 'GNWT', 'HOT', 'RPT']:
        if theory in dogmatism_by_theory.index:
            row = dogmatism_by_theory.loc[theory]
            print(f"  {theory}: {row['mean']:.2f} ± {row['std']:.2f} (n={int(row['count'])})")
            results[theory] = {
                'mean': float(row['mean']),
                'std': float(row['std']),
                'n': int(row['count'])
            }

    # Statistical test: Is there significant difference?
    theory_groups = [theories_df[theories_df['theory'] == t]['dogmatism_score'].values
                     for t in ['IIT', 'GNWT', 'HOT', 'RPT']
                     if len(theories_df[theories_df['theory'] == t]) >= 5]

    if len(theory_groups) >= 2:
        f_stat, p_val = stats.f_oneway(*theory_groups)
        print(f"\n  ANOVA: F={f_stat:.3f}, p={p_val:.4f} {'✅ SIGNIFICANT' if p_val < 0.05 else '❌ Not significant'}")
        results['anova'] = {'f_statistic': float(f_stat), 'p_value': float(p_val), 'significant': p_val < 0.05}

    # Plot dogmatism by theory
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    theories = ['IIT', 'GNWT', 'HOT', 'RPT']
    means = [dogmatism_by_theory.loc[t, 'mean'] if t in dogmatism_by_theory.index else 0 for t in theories]
    stds = [dogmatism_by_theory.loc[t, 'std'] if t in dogmatism_by_theory.index else 0 for t in theories]
    colors = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A']

    bars = axes[0].bar(theories, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Mean Dogmatism Score (1-10)', fontsize=11)
    axes[0].set_title('Epistemic Confidence by Theory', fontsize=12)
    axes[0].set_ylim(1, 10)
    axes[0].axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='Baseline (neutral)')

    for bar, mean in zip(bars, means):
        if mean > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f'{mean:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Box plot
    box_data = [theories_df[theories_df['theory'] == t]['dogmatism_score'].values for t in theories]
    bp = axes[1].boxplot(box_data, labels=theories, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel('Dogmatism Score Distribution', fontsize=11)
    axes[1].set_title('Dogmatism Distribution by Theory', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "12_dogmatism_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: 12_dogmatism_analysis.png")

    # Dogmatism over time
    yearly_dogmatism = theories_df.groupby(['Year', 'theory'])['dogmatism_score'].mean().unstack(fill_value=np.nan)
    yearly_dogmatism = yearly_dogmatism[(yearly_dogmatism.index >= MIN_YEAR) & (yearly_dogmatism.index <= MAX_YEAR)]

    if len(yearly_dogmatism) > 3:
        plt.figure(figsize=(12, 6))
        for theory in ['IIT', 'GNWT', 'HOT', 'RPT']:
            if theory in yearly_dogmatism.columns:
                data = yearly_dogmatism[theory].dropna()
                if len(data) > 2:
                    plt.plot(data.index, data.values, marker='o', label=theory,
                            color=colors[theories.index(theory)], linewidth=2, markersize=6)

        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Mean Dogmatism Score', fontsize=12)
        plt.title('Epistemic Confidence Over Time by Theory', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "12b_dogmatism_over_time.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved: 12b_dogmatism_over_time.png")

    return results

# =============================================================================
# CELL 14: METHODOLOGY PREDICTS THEORY (Yaron et al. 2022)
# =============================================================================

def run_methodology_predicts_theory(df: pd.DataFrame, output_folder: str) -> Dict:
    """
    Replicate Yaron et al. (2022) ConTraSt finding: methodology choices predict theory.
    Uses Random Forest to predict theory from paradigm, type, and subject.
    """
    print(f"\n{'='*60}\n🔬 METHODOLOGY PREDICTS THEORY (Yaron et al. 2022)\n{'='*60}")

    # Filter to classified theories
    theories_df = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])].copy()

    if len(theories_df) < 50:
        print("  ⚠️ Insufficient theory-classified papers")
        return {"error": "Insufficient data"}

    # Encode features
    le_paradigm = LabelEncoder()
    le_type = LabelEncoder()
    le_subject = LabelEncoder()
    le_target = LabelEncoder()
    le_theory = LabelEncoder()

    theories_df['paradigm_enc'] = le_paradigm.fit_transform(theories_df['paradigm'])
    theories_df['type_enc'] = le_type.fit_transform(theories_df['type'])
    theories_df['subject_enc'] = le_subject.fit_transform(theories_df['subject'])
    theories_df['target_enc'] = le_target.fit_transform(theories_df['target'])
    theories_df['theory_enc'] = le_theory.fit_transform(theories_df['theory'])

    # Features and target
    feature_cols = ['paradigm_enc', 'type_enc', 'subject_enc', 'target_enc']
    X = theories_df[feature_cols].values
    y = theories_df['theory_enc'].values

    # Random Forest with cross-validation
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')

    print(f"  Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit full model for feature importances
    rf.fit(X, y)

    feature_names = ['Paradigm', 'Type', 'Subject', 'Target']
    importances = rf.feature_importances_

    print("\n  Feature Importances:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"    {name}: {imp:.3f}")

    results = {
        'cv_accuracy_mean': float(cv_scores.mean()),
        'cv_accuracy_std': float(cv_scores.std()),
        'feature_importances': dict(zip(feature_names, [float(i) for i in importances])),
        'n_samples': len(theories_df)
    }

    # Confusion matrix
    y_pred = rf.predict(X)
    cm = confusion_matrix(y, y_pred)
    theory_labels = le_theory.classes_

    if MATPLOTLIB_AVAILABLE:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Feature importance bar plot
        sorted_idx = np.argsort(importances)[::-1]
        axes[0].barh([feature_names[i] for i in sorted_idx],
                    [importances[i] for i in sorted_idx],
                    color=['#E63946', '#457B9D', '#2A9D8F', '#E9C46A'])
        axes[0].set_xlabel('Feature Importance', fontsize=11)
        axes[0].set_title(f'Methodology Predicts Theory\n(CV Accuracy: {cv_scores.mean():.1%})', fontsize=12)

        # Confusion matrix heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=theory_labels, yticklabels=theory_labels, ax=axes[1])
        axes[1].set_xlabel('Predicted', fontsize=11)
        axes[1].set_ylabel('Actual', fontsize=11)
        axes[1].set_title('Confusion Matrix', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "10_methodology_predicts_theory.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved: 10_methodology_predicts_theory.png")

    # Detailed crosstabs
    print("\n  Paradigm × Theory crosstab:")
    ct_paradigm = pd.crosstab(theories_df['theory'], theories_df['paradigm'], normalize='index') * 100
    print(ct_paradigm.round(1).to_string())

    results['crosstab_paradigm'] = ct_paradigm.to_dict()

    return results

# =============================================================================
# CELL 15: ANALYSES 1-7 (IMPROVED)
# =============================================================================

def run_contrast_audit(df: pd.DataFrame, output_folder: str) -> Dict:
    """Analysis 1: ConTraSt Sankey diagram."""
    print(f"\n{'='*60}\n📊 ANALYSIS 1: CONTRAST SANKEY\n{'='*60}")

    theories_df = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])].copy()
    if len(theories_df) < 20:
        print("  ⚠️ Insufficient data")
        return {"error": "Insufficient data"}

    # Sankey: Paradigm -> Theory
    counts = theories_df.groupby(['paradigm', 'theory']).size().reset_index(name='count')

    paradigms = theories_df['paradigm'].unique().tolist()
    theories = theories_df['theory'].unique().tolist()
    all_labels = paradigms + theories

    source_idx = [paradigms.index(r['paradigm']) for _, r in counts.iterrows()]
    target_idx = [len(paradigms) + theories.index(r['theory']) for _, r in counts.iterrows()]

    colors_source = ['#888888'] * len(paradigms)
    colors_target = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A'][:len(theories)]

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=all_labels,
            color=colors_source + colors_target
        ),
        link=dict(
            source=source_idx,
            target=target_idx,
            value=counts['count'].tolist()
        )
    ))

    fig.update_layout(title="ConTraSt: Paradigm → Theory Flow", font_size=12)
    fig.write_image(os.path.join(output_folder, "1_contrast.png"), scale=2)
    print("  ✅ Saved: 1_contrast.png")

    return {"n_papers": len(theories_df)}


def run_moving_goalposts(df: pd.DataFrame, output_folder: str) -> Dict:
    """Analysis 2: Semantic drift over time."""
    print(f"\n{'='*60}\n📈 ANALYSIS 2: MOVING GOALPOSTS\n{'='*60}")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    theories_df = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])].copy()
    theories_df['text'] = theories_df['Abstract'].apply(lambda x: extract_first_sentences(x, 3))

    colors = {'IIT': '#E63946', 'GNWT': '#457B9D', 'HOT': '#2A9D8F', 'RPT': '#E9C46A'}
    fig = go.Figure()

    for theory in ['IIT', 'GNWT', 'HOT', 'RPT']:
        theory_df = theories_df[theories_df['theory'] == theory]
        if len(theory_df) < 5:
            continue

        # Get embeddings by year
        yearly_embeddings = {}
        for year in range(MIN_YEAR, MAX_YEAR + 1):
            year_texts = theory_df[theory_df['Year'] == year]['text'].tolist()
            if len(year_texts) >= 2:
                yearly_embeddings[year] = model.encode(year_texts)

        if len(yearly_embeddings) < 3:
            continue

        years = sorted(yearly_embeddings.keys())
        baseline_year = BASELINE_YEAR if BASELINE_YEAR in years else years[0]
        baseline_emb = np.mean(yearly_embeddings[baseline_year], axis=0)

        drift_data = []
        for year in years:
            year_centroid = np.mean(yearly_embeddings[year], axis=0)
            drift = cosine(baseline_emb, year_centroid)
            drift_data.append({'year': year, 'drift': drift})

        drift_df = pd.DataFrame(drift_data)
        fig.add_trace(go.Scatter(
            x=drift_df['year'], y=drift_df['drift'],
            mode='lines+markers', name=theory,
            line=dict(color=colors[theory], width=2),
            marker=dict(size=8)
        ))

    fig.update_layout(
        title=f"Semantic Drift from {BASELINE_YEAR} Baseline",
        xaxis_title="Year", yaxis_title="Cosine Distance from Baseline",
        legend=dict(x=0.02, y=0.98)
    )
    fig.write_image(os.path.join(output_folder, "2_goalposts.png"), scale=2)
    print("  ✅ Saved: 2_goalposts.png")

    return {}


def run_adversarial(df: pd.DataFrame, output_folder: str) -> Dict:
    """Analysis 3: Cross-theory engagement (adversarial index)."""
    print(f"\n{'='*60}\n⚔️ ANALYSIS 3: ADVERSARIAL INDEX\n{'='*60}")

    patterns = {
        'IIT': r'\b(integrated information|iit\b|tononi|phi\b)',
        'GNWT': r'\b(global.?workspace|gnw\b|dehaene|baars)',
        'HOT': r'\b(higher.?order|rosenthal|hot\b)',
        'RPT': r'\b(recurrent processing|lamme|rpt\b)'
    }

    def count_theories(abstract):
        if pd.isna(abstract):
            return 0
        text = str(abstract).lower()
        return sum(1 for p in patterns.values() if re.search(p, text))

    df['theories_mentioned'] = df['Abstract'].apply(count_theories)
    df['adversarial'] = df['theories_mentioned'] >= 2

    yearly = df.groupby('Year').agg(
        total=('Title', 'count'),
        adversarial=('adversarial', 'sum')
    ).reset_index()
    yearly['pct'] = yearly['adversarial'] / yearly['total'] * 100
    yearly = yearly[(yearly['Year'] >= MIN_YEAR) & (yearly['Year'] <= MAX_YEAR)]

    if len(yearly) < 3:
        return {"error": "Insufficient data"}

    z = np.polyfit(yearly['Year'], yearly['pct'], 1)
    trend = "↑ INCREASING" if z[0] > 0.1 else "↓ DECREASING" if z[0] < -0.1 else "→ STABLE"

    print(f"  Total adversarial papers: {df['adversarial'].sum()}")
    print(f"  Trend: {trend} ({z[0]:.3f}%/year)")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly['Year'], y=yearly['pct'], name='Adversarial %',
                         marker_color='#E63946', opacity=0.7))
    fig.add_trace(go.Scatter(x=yearly['Year'], y=np.poly1d(z)(yearly['Year']),
                             mode='lines', name='Trend', line=dict(color='black', dash='dash', width=2)))

    fig.update_layout(title=f"Adversarial Index (Trend: {trend})",
                     xaxis_title="Year", yaxis_title="% Papers Mentioning 2+ Theories")
    fig.write_image(os.path.join(output_folder, "3_adversarial.png"), scale=2)
    print("  ✅ Saved: 3_adversarial.png")

    return {"total_adversarial": int(df['adversarial'].sum()), "trend_slope": float(z[0])}


def run_cartography(df: pd.DataFrame, output_folder: str) -> Dict:
    """Analysis 4: Brain region preferences by theory."""
    print(f"\n{'='*60}\n🧠 ANALYSIS 4: NEURO-CARTOGRAPHY\n{'='*60}")

    theories_df = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])]
    if len(theories_df) < 10:
        return {"error": "Insufficient data"}

    region_counts = defaultdict(lambda: defaultdict(int))
    for _, row in theories_df.iterrows():
        anatomy = row.get('anatomy', [])
        if isinstance(anatomy, list):
            for region in anatomy:
                if region and str(region).lower() not in ['none', 'null', '']:
                    region_counts[row['theory']][str(region).upper()] += 1

    # Get top regions
    all_regions = Counter()
    for theory_regions in region_counts.values():
        all_regions.update(theory_regions)

    top_regions = [r for r, _ in all_regions.most_common(15)]

    if not top_regions:
        print("  ⚠️ No brain regions extracted")
        return {"error": "No regions found"}

    # Build matrix
    matrix = []
    for theory in ['IIT', 'GNWT', 'HOT', 'RPT']:
        row = [region_counts[theory].get(r, 0) for r in top_regions]
        total = sum(row)
        row = [x / total * 100 if total > 0 else 0 for x in row]
        matrix.append(row)

    fig = go.Figure(go.Heatmap(
        z=matrix, x=top_regions, y=['IIT', 'GNWT', 'HOT', 'RPT'],
        colorscale='RdBu', reversescale=True
    ))
    fig.update_layout(title="Neuro-Cartography: Brain Regions by Theory")
    fig.write_image(os.path.join(output_folder, "4_cartography.png"), scale=2)
    print("  ✅ Saved: 4_cartography.png")

    return {}


def run_aggression(df: pd.DataFrame, output_folder: str) -> Dict:
    """Analysis 5: Hostile tone over time."""
    print(f"\n{'='*60}\n😤 ANALYSIS 5: AGGRESSION INDEX\n{'='*60}")

    theories_df = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])]
    if len(theories_df) < 10:
        return {"error": "Insufficient data"}

    # Calculate aggressive tone percentage
    def is_aggressive(tone):
        return tone in ['Dismissive', 'Critical']

    theories_df = theories_df.copy()
    theories_df['aggressive'] = theories_df['tone'].apply(is_aggressive)

    yearly = theories_df.groupby('Year').agg(
        total=('Title', 'count'),
        aggressive=('aggressive', 'sum')
    ).reset_index()
    yearly['pct'] = yearly['aggressive'] / yearly['total'] * 100
    yearly = yearly[(yearly['Year'] >= MIN_YEAR) & (yearly['Year'] <= MAX_YEAR)]

    if len(yearly) < 3:
        return {"error": "Insufficient data"}

    # Trend
    z = np.polyfit(yearly['Year'], yearly['pct'], 1)
    trend = "↑ INCREASING" if z[0] > 0.1 else "↓ DECREASING" if z[0] < -0.1 else "→ STABLE"

    print(f"  Aggressive papers: {theories_df['aggressive'].sum()} / {len(theories_df)}")
    print(f"  Trend: {trend} ({z[0]:.3f}%/year)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly['Year'], y=yearly['pct'], mode='lines+markers',
                             name='Aggression %', line=dict(color='#E63946', width=2)))
    fig.add_trace(go.Scatter(x=yearly['Year'], y=np.poly1d(z)(yearly['Year']),
                             mode='lines', name='Trend', line=dict(color='black', dash='dash')))

    fig.update_layout(title=f"Aggression Index (Trend: {trend})",
                     xaxis_title="Year", yaxis_title="% Papers with Critical/Dismissive Tone")
    fig.write_image(os.path.join(output_folder, "5_aggression.png"), scale=2)
    print("  ✅ Saved: 5_aggression.png")

    return {"total_aggressive": int(theories_df['aggressive'].sum()), "trend": trend}


def run_easy_hard(df: pd.DataFrame, output_folder: str) -> Dict:
    """Analysis 6: Easy vs Hard problem focus."""
    print(f"\n{'='*60}\n🤔 ANALYSIS 6: EASY VS HARD PROBLEM\n{'='*60}")

    theories_df = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])]
    if len(theories_df) < 10:
        return {"error": "Insufficient data"}

    crosstab = pd.crosstab(theories_df['theory'], theories_df['target'], normalize='index') * 100

    for col in ['Phenomenology', 'Function', 'Mechanism']:
        if col not in crosstab.columns:
            crosstab[col] = 0

    print("  Target focus by theory (%):")
    print(crosstab[['Phenomenology', 'Function', 'Mechanism']].round(1).to_string())

    fig = go.Figure()
    colors = {'Phenomenology': '#E63946', 'Function': '#457B9D', 'Mechanism': '#2A9D8F'}

    for target in ['Phenomenology', 'Function', 'Mechanism']:
        fig.add_trace(go.Bar(
            x=crosstab.index,
            y=crosstab[target],
            name=target,
            marker_color=colors[target]
        ))

    fig.update_layout(
        title="Easy vs Hard Problem: Target Focus by Theory",
        xaxis_title="Theory", yaxis_title="% of Papers",
        barmode='stack'
    )
    fig.write_image(os.path.join(output_folder, "6_easy_hard.png"), scale=2)
    print("  ✅ Saved: 6_easy_hard.png")

    return crosstab.to_dict()


def run_organism(df: pd.DataFrame, output_folder: str) -> Dict:
    """Analysis 7: Model organism preferences."""
    print(f"\n{'='*60}\n🐁 ANALYSIS 7: MODEL ORGANISM\n{'='*60}")

    theories_df = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])]
    if len(theories_df) < 10:
        return {"error": "Insufficient data"}

    crosstab = pd.crosstab(theories_df['theory'], theories_df['subject'], normalize='index') * 100

    print("  Subject by theory (%):")
    print(crosstab.round(1).to_string())

    fig = go.Figure()
    colors = {'Human': '#457B9D', 'Clinical': '#E63946', 'Animal': '#2A9D8F',
              'Simulation': '#E9C46A', 'Review': '#888888'}

    for subject in ['Human', 'Clinical', 'Animal', 'Simulation', 'Review']:
        if subject in crosstab.columns:
            fig.add_trace(go.Bar(
                x=crosstab.index,
                y=crosstab[subject],
                name=subject,
                marker_color=colors.get(subject, '#888888')
            ))

    fig.update_layout(
        title="Model Organism Preferences by Theory",
        xaxis_title="Theory", yaxis_title="% of Papers",
        barmode='stack'
    )
    fig.write_image(os.path.join(output_folder, "7_organism.png"), scale=2)
    print("  ✅ Saved: 7_organism.png")

    return crosstab.to_dict()

# =============================================================================
# CELL 16: MAIN PIPELINE
# =============================================================================

def run_full_audit(csv_path: str = CSV_PATH, api_key: str = GEMINI_API_KEY,
                   output_folder: str = OUTPUT_FOLDER, use_llm: bool = True,
                   use_rules_only: bool = False, run_embeddings: bool = True) -> Dict:
    """
    Run the full consciousness wars audit pipeline.
    """
    print(f"\n{'='*70}")
    print("    🧠 CONSCIOUSNESS WARS BIBLIOMETRIC AUDIT v4.0 🧠")
    print(f"{'='*70}")

    os.makedirs(output_folder, exist_ok=True)

    # Load data
    df = load_bibliography(csv_path)

    # Extract dimensions
    if use_rules_only:
        df_classified = RuleBasedExtractor().batch_extract(df)
    elif use_llm and api_key:
        try:
            extractor = GeminiExtractor(api_key, GEMINI_MODEL)
            # Test connection
            test_result = extractor.extract_classification("Integrated information theory and phi measures.")
            print(f"✅ Gemini API connected | Test: {test_result['theory']} (conf: {test_result['confidence']:.2f})")
            df_classified = extractor.batch_extract_parallel(df, MAX_WORKERS)
        except Exception as e:
            print(f"⚠️ Gemini API failed: {e}")
            print("Falling back to rule-based extraction...")
            df_classified = RuleBasedExtractor().batch_extract(df)
    else:
        df_classified = RuleBasedExtractor().batch_extract(df)

    results = {}

    # Run analyses 1-7
    analyses = [
        ('1_contrast', run_contrast_audit),
        ('2_goalposts', run_moving_goalposts),
        ('3_adversarial', run_adversarial),
        ('4_cartography', run_cartography),
        ('5_aggression', run_aggression),
        ('6_easy_hard', run_easy_hard),
        ('7_organism', run_organism),
    ]

    for name, func in analyses:
        try:
            results[name] = func(df_classified, output_folder)
        except Exception as e:
            print(f"⚠️ {name} failed: {e}")
            results[name] = {"error": str(e)}

    # Run methodology predicts theory (Analysis 10)
    try:
        results['10_methodology'] = run_methodology_predicts_theory(df_classified, output_folder)
    except Exception as e:
        print(f"⚠️ Methodology analysis failed: {e}")

    # Run dogmatism analysis (Analysis 12)
    try:
        results['12_dogmatism'] = run_dogmatism_analysis(df_classified, output_folder)
    except Exception as e:
        print(f"⚠️ Dogmatism analysis failed: {e}")

    # Run embedding-based analyses
    if run_embeddings and api_key:
        try:
            embedder = GeminiEmbedder(api_key, EMBEDDING_MODEL)
            embeddings = embedder.embed_batch(df_classified['Abstract'].tolist(), EMBEDDING_BATCH_SIZE)

            # Semantic clustering (Analysis 8)
            cluster_result = run_semantic_clustering(df_classified, embeddings, output_folder)

            if 'error' not in cluster_result:
                # Hypothesis tests (Analysis 9)
                results['9_hypothesis'] = run_hypothesis_tests(
                    cluster_result['valid_df'],
                    cluster_result['centroids'],
                    cluster_result['X'],
                    output_folder
                )

                # Vocabulary divergence (Analysis 11)
                results['11_divergence'] = run_vocabulary_divergence(
                    df_classified, embeddings, output_folder
                )
        except Exception as e:
            print(f"⚠️ Embedding-based analyses failed: {e}")

    # Save results with error handling
    csv_path = os.path.join(output_folder, "classified_papers.csv")
    try:
        df_classified.to_csv(csv_path, index=False)
    except PermissionError:
        # File might be open - try alternative name
        alt_path = os.path.join(output_folder, f"classified_papers_{int(time.time())}.csv")
        print(f"⚠️ Cannot write to {csv_path} (file in use). Saving to {alt_path}")
        df_classified.to_csv(alt_path, index=False)
        csv_path = alt_path

    json_path = os.path.join(output_folder, "results.json")
    try:
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except PermissionError:
        alt_path = os.path.join(output_folder, f"results_{int(time.time())}.json")
        print(f"⚠️ Cannot write to {json_path}. Saving to {alt_path}")
        with open(alt_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Final summary
    print(f"\n{'='*70}")
    print("    ✅ AUDIT COMPLETE")
    print(f"{'='*70}")
    print(f"Total papers processed: {len(df_classified)}")
    print(f"\n📊 Final Theory Distribution:")
    for theory, count in df_classified['theory'].value_counts().items():
        print(f"   {theory}: {count} ({count/len(df_classified)*100:.1f}%)")

    print(f"\n📊 Extraction Quality:")
    failed = df_classified['_extraction_failed'].sum()
    print(f"   Successful extractions: {len(df_classified) - failed} ({(1 - failed/len(df_classified))*100:.1f}%)")
    print(f"   Mean confidence: {df_classified['confidence'].mean():.3f}")
    print(f"   Mean dogmatism: {df_classified['dogmatism_score'].mean():.2f}")

    print(f"\n📁 Results saved to: {output_folder}")

    return {'data': df_classified, 'results': results}


def test_api_connection(api_key: str, model: str = GEMINI_MODEL) -> bool:
    """Test the Gemini API connection with a simple request."""
    print(f"\n{'='*60}")
    print("🔌 TESTING API CONNECTION")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"API Key: {'*' * 20}{api_key[-4:] if len(api_key) > 4 else '???'}")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": "Reply with just the word 'OK' if you can read this."}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 10}
    }

    try:
        r = requests.post(f"{url}?key={api_key}", json=payload, timeout=30)
        print(f"HTTP Status: {r.status_code}")

        if r.status_code == 200:
            result = r.json()
            if 'candidates' in result:
                text = result['candidates'][0].get('content', {}).get('parts', [{}])[0].get('text', '')
                print(f"Response: {text[:50]}")
                print("✅ API connection successful!")
                return True
            else:
                print(f"⚠️ Unexpected response: {str(result)[:200]}")
        else:
            print(f"❌ API Error: {r.text[:300]}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

    return False


if __name__ == "__main__":
    print("=" * 70)
    print("    CONSCIOUSNESS WARS BIBLIOMETRIC AUDIT v4.0")
    print("=" * 70)

    # Check API key
    if not GEMINI_API_KEY:
        print("\n❌ ERROR: GEMINI_API_KEY not found!")
        print("Please create a .env file with: GEMINI_API_KEY=your_key_here")
        exit(1)

    print(f"\n📁 CSV Path: {CSV_PATH}")
    print(f"📁 Output Folder: {OUTPUT_FOLDER}")
    print(f"🤖 Model: {GEMINI_MODEL}")

    # Test API connection first
    if not test_api_connection(GEMINI_API_KEY, GEMINI_MODEL):
        print("\n❌ API connection failed. Please check your API key and model name.")
        print("Available models: gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro")
        exit(1)

    # Run the full audit
    print("\n" + "=" * 70)
    print("    STARTING FULL AUDIT")
    print("=" * 70)

    results = run_full_audit()

    print("\n✅ Audit complete!")
