"""
================================================================================
CONSCIOUSNESS WARS BIBLIOMETRIC AUDIT PIPELINE v2.0
================================================================================
A comprehensive bibliometric analysis of consciousness theories:
IIT (Integrated Information Theory), GNWT (Global Neuronal Workspace Theory),
HOT (Higher-Order Theory), and RPT (Recurrent Processing Theory).

Seven Analyses:
1. ConTraSt Audit - Replication of Yaron et al. (2022)
2. Moving Goalposts - Concept drift analysis over time
3. Adversarial Index - Cross-theory citation patterns
4. Neuro-Cartography - Brain region biases per theory
5. Aggression Index - Hostility trends over time
6. Easy vs Hard Problem - Phenomenology vs Function
7. Model Organism Bias - Subject/species distribution

Extracts 8 dimensions per paper using Gemini LLM.
================================================================================
"""

# =============================================================================
# CELL 1: CONFIGURATION - EDIT THIS SECTION
# =============================================================================

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        🔧 CONFIGURATION 🔧                                 ║
# ╠═══════════════════════════════════════════════════════════════════════════╣
# ║  Set your API key and output folder below before running                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Your Gemini API Key (get one at https://aistudio.google.com/app/apikey)
GEMINI_API_KEY = ""  # <-- PASTE YOUR API KEY HERE

# Output folder for saving results
OUTPUT_FOLDER = "./results"  # Local folder

# Path to your bibliography CSV file
CSV_PATH = "cons_bib.csv"  # Update if your file is elsewhere

# Gemini model to use
GEMINI_MODEL = "gemini-2.0-flash"  # or "gemini-1.5-flash", "gemini-1.5-pro"

# Parallel processing settings
MAX_WORKERS = 5       # Number of parallel API requests (lower = more reliable)
API_DELAY = 0.2       # Delay between requests (seconds)
MAX_RETRIES = 3       # Retries per failed request

# Analysis settings
BASELINE_YEAR = 2005  # Reference year for semantic drift calculation
MIN_YEAR = 2000       # Earliest year to include in analysis
MAX_YEAR = 2025       # Latest year to include in analysis

# Debug mode - set to True to see raw API responses
DEBUG_MODE = False
DEBUG_SAMPLE_SIZE = 5  # Number of raw responses to print

# =============================================================================
# CELL 2: INSTALLATION (Run once)
# =============================================================================

def install_packages():
    """Install required packages for the analysis."""
    import subprocess
    import sys

    packages = [
        'pandas',
        'numpy',
        'plotly',
        'scikit-learn',
        'sentence-transformers',
        'requests',
        'tqdm',
        'kaleido',  # For PNG export
    ]

    print("=" * 60)
    print("📦 INSTALLING REQUIRED PACKAGES")
    print("=" * 60)

    for package in packages:
        print(f"  Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

    print("\n✅ All packages installed successfully!")
    print("=" * 60)

# Uncomment to install (run once):
# install_packages()

# =============================================================================
# CELL 3: IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import json
import re
import time
import os
import warnings
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

import requests
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# CELL 4: DATA LOADING
# =============================================================================

def load_bibliography(csv_path: str) -> pd.DataFrame:
    """Load bibliography data from CSV file."""
    print("\n" + "=" * 60)
    print("📚 LOADING BIBLIOGRAPHY DATA")
    print("=" * 60)

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()

        print(f"✅ Loaded {len(df)} papers")
        print(f"📅 Year range: {df['Year'].min()} - {df['Year'].max()}")

        # Filter papers with valid abstracts
        df_valid = df[df['Abstract'].notna() & (df['Abstract'].str.len() > 50)].copy()
        print(f"📄 Papers with valid abstracts: {len(df_valid)}")

        # Convert year
        df_valid['Year'] = pd.to_numeric(df_valid['Year'], errors='coerce')
        df_valid = df_valid[df_valid['Year'].notna()]
        df_valid['Year'] = df_valid['Year'].astype(int)

        return df_valid.reset_index(drop=True)

    except FileNotFoundError:
        print(f"❌ Error: File not found at {csv_path}")
        raise

def extract_first_sentences(text: str, n_sentences: int = 2) -> str:
    """Extract the first n sentences from text."""
    if pd.isna(text) or not text:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return ' '.join(sentences[:n_sentences])

# =============================================================================
# CELL 5: GEMINI LLM EXTRACTOR - FIXED VERSION
# =============================================================================

class GeminiExtractor:
    """
    Extract structured information from abstracts using Google Gemini API.
    Extracts 8 dimensions per paper with robust error handling.
    """

    # Simplified, clearer prompt for better extraction
    EXTRACTION_PROMPT = '''Analyze this consciousness research abstract and classify it on 8 dimensions.

ABSTRACT:
{abstract}

CLASSIFICATION GUIDE:

1. THEORY - Primary consciousness theory:
   - IIT: Mentions integrated information, phi (Φ), Tononi, information integration, exclusion postulate
   - GNWT: Mentions global workspace, Dehaene, Baars, ignition, broadcasting, access consciousness
   - HOT: Mentions higher-order thought/theory, Rosenthal, metacognition, higher-order representation
   - RPT: Mentions recurrent processing, Lamme, feedback loops, recurrent activity
   - Neutral: No clear theoretical alignment, general consciousness study, or methodology paper

2. PARADIGM - Experimental approach:
   - Report: Participants report their experience (verbal, button press, ratings)
   - No-Report: No explicit reports required (passive viewing, physiological measures only)

3. TYPE - What aspect studied:
   - Content: What we perceive (visual features, objects, specific experiences)
   - State: Level/presence of consciousness (sleep, anesthesia, coma, awareness levels)

4. EPISTEMIC - Hypothesis approach:
   - A Priori: Theory-driven predictions made before data collection
   - Post-Hoc: Exploratory analysis, interpretations after seeing results

5. ANATOMY - Top 3 brain regions mentioned (use standard abbreviations):
   Examples: PFC, ACC, PPC, V1, V4, IT, MTL, Thalamus, Claustrum, Insula, TPJ
   Use ["None"] if no specific regions mentioned.

6. TONE - Attitude toward rival theories (if mentioned):
   - Dismissive: Rejects other theories as fundamentally flawed
   - Critical: Points out specific problems with other theories
   - Constructive: Seeks integration or acknowledges complementary aspects
   - Neutral: No evaluation of other theories or purely descriptive

7. TARGET - What the paper explains:
   - Phenomenology: Subjective experience, qualia, what-it-is-like, the hard problem
   - Function: Access, reportability, cognitive functions, information processing
   - Mechanism: Neural mechanisms, correlates, implementation details

8. SUBJECT - Study population:
   - Human: Healthy human participants
   - Clinical: Patients (DOC, anesthesia, neurological conditions)
   - Animal: Non-human animals (primates, rodents, etc.)
   - Simulation: Computational models, simulations, theoretical
   - Review: Literature review, meta-analysis, theoretical review

Respond with ONLY valid JSON (no markdown, no explanation):
{{"theory":"IIT","paradigm":"Report","type":"Content","epistemic":"A Priori","anatomy":["PFC","Thalamus"],"tone":"Neutral","target":"Function","subject":"Human"}}'''

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.session = requests.Session()
        self.debug_responses = []

        if not api_key or api_key == "":
            raise ValueError("❌ GEMINI_API_KEY is not set!")

    def _call_gemini(self, abstract: str) -> Tuple[Dict[str, Any], str]:
        """Make a single Gemini API call. Returns (parsed_result, raw_response)."""
        prompt = self.EXTRACTION_PROMPT.format(abstract=abstract[:2500])

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,  # Deterministic
                "maxOutputTokens": 200,
                "topP": 1.0,
            }
        }

        try:
            response = self.session.post(
                f"{self.base_url}?key={self.api_key}",
                json=payload,
                timeout=30
            )

            if response.status_code == 429:
                time.sleep(3)
                return {"_retry": True}, "RATE_LIMITED"

            if response.status_code != 200:
                return self._default(), f"HTTP_{response.status_code}"

            result = response.json()

            # Extract text from response
            if 'candidates' in result and result['candidates']:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    raw_text = candidate['content']['parts'][0].get('text', '')
                    parsed = self._parse_response(raw_text)
                    return parsed, raw_text

            return self._default(), "NO_CANDIDATES"

        except requests.exceptions.Timeout:
            return self._default(), "TIMEOUT"
        except requests.exceptions.RequestException as e:
            return self._default(), f"REQUEST_ERROR: {str(e)[:50]}"
        except Exception as e:
            return self._default(), f"ERROR: {str(e)[:50]}"

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from Gemini response with multiple fallback strategies."""
        if not text or len(text) < 10:
            return self._default()

        # Clean the text
        text = text.strip()

        # Remove markdown code blocks
        text = re.sub(r'^```json\s*\n?', '', text)
        text = re.sub(r'^```\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
        text = text.strip()

        # Try direct JSON parse first
        try:
            data = json.loads(text)
            return self._validate_and_normalize(data)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object with regex
        json_patterns = [
            r'\{[^{}]*"theory"[^{}]*\}',  # Simple object with theory
            r'\{[\s\S]*?"theory"[\s\S]*?\}',  # Multiline
        ]

        for pattern in json_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    return self._validate_and_normalize(data)
                except json.JSONDecodeError:
                    continue

        # Last resort: extract individual fields with regex
        return self._extract_fields_from_text(text)

    def _validate_and_normalize(self, data: Dict) -> Dict[str, Any]:
        """Validate and normalize parsed JSON data."""
        result = self._default()

        # Theory
        theory = str(data.get("theory", "")).upper().strip()
        if theory in ["IIT", "GNWT", "HOT", "RPT"]:
            result["theory"] = theory
        elif "NEUTRAL" in theory or not theory:
            result["theory"] = "Neutral"

        # Paradigm
        paradigm = str(data.get("paradigm", "")).lower()
        result["paradigm"] = "No-Report" if "no" in paradigm else "Report"

        # Type
        type_val = str(data.get("type", "")).lower()
        result["type"] = "State" if "state" in type_val else "Content"

        # Epistemic
        epistemic = str(data.get("epistemic", "")).lower()
        result["epistemic"] = "A Priori" if "priori" in epistemic else "Post-Hoc"

        # Anatomy - handle both list and string
        anatomy = data.get("anatomy", [])
        if isinstance(anatomy, str):
            anatomy = [a.strip() for a in anatomy.split(",") if a.strip()]
        elif isinstance(anatomy, list):
            anatomy = [str(a).strip() for a in anatomy if a and str(a).strip().lower() != "none"]
        result["anatomy"] = anatomy[:3] if anatomy else []

        # Tone
        tone = str(data.get("tone", "")).lower()
        if "dismiss" in tone:
            result["tone"] = "Dismissive"
        elif "critic" in tone:
            result["tone"] = "Critical"
        elif "construct" in tone:
            result["tone"] = "Constructive"
        else:
            result["tone"] = "Neutral"

        # Target
        target = str(data.get("target", "")).lower()
        if "phenom" in target:
            result["target"] = "Phenomenology"
        elif "mechan" in target:
            result["target"] = "Mechanism"
        else:
            result["target"] = "Function"

        # Subject
        subject = str(data.get("subject", "")).lower()
        if "clinical" in subject or "patient" in subject:
            result["subject"] = "Clinical"
        elif "animal" in subject:
            result["subject"] = "Animal"
        elif "simul" in subject or "model" in subject or "comput" in subject:
            result["subject"] = "Simulation"
        elif "review" in subject or "meta" in subject:
            result["subject"] = "Review"
        else:
            result["subject"] = "Human"

        result["confidence"] = 0.8
        return result

    def _extract_fields_from_text(self, text: str) -> Dict[str, Any]:
        """Fallback: extract fields from unstructured text."""
        result = self._default()
        text_upper = text.upper()

        # Theory - look for explicit mentions
        for theory in ["IIT", "GNWT", "HOT", "RPT"]:
            # Look for patterns like "theory": "IIT" or theory: IIT
            if re.search(rf'["\']?theory["\']?\s*:\s*["\']?{theory}', text, re.IGNORECASE):
                result["theory"] = theory
                break

        # Other fields
        if re.search(r'["\']?paradigm["\']?\s*:\s*["\']?no', text, re.IGNORECASE):
            result["paradigm"] = "No-Report"

        if re.search(r'["\']?type["\']?\s*:\s*["\']?state', text, re.IGNORECASE):
            result["type"] = "State"

        if re.search(r'["\']?epistemic["\']?\s*:\s*["\']?a\s*priori', text, re.IGNORECASE):
            result["epistemic"] = "A Priori"

        if re.search(r'["\']?tone["\']?\s*:\s*["\']?dismiss', text, re.IGNORECASE):
            result["tone"] = "Dismissive"
        elif re.search(r'["\']?tone["\']?\s*:\s*["\']?critic', text, re.IGNORECASE):
            result["tone"] = "Critical"
        elif re.search(r'["\']?tone["\']?\s*:\s*["\']?construct', text, re.IGNORECASE):
            result["tone"] = "Constructive"

        if re.search(r'["\']?target["\']?\s*:\s*["\']?phenom', text, re.IGNORECASE):
            result["target"] = "Phenomenology"
        elif re.search(r'["\']?target["\']?\s*:\s*["\']?mechan', text, re.IGNORECASE):
            result["target"] = "Mechanism"

        result["confidence"] = 0.3
        return result

    def _default(self) -> Dict[str, Any]:
        """Return default extraction."""
        return {
            "theory": "Neutral",
            "paradigm": "Report",
            "type": "Content",
            "epistemic": "Post-Hoc",
            "anatomy": [],
            "tone": "Neutral",
            "target": "Function",
            "subject": "Human",
            "confidence": 0.0
        }

    def extract_single(self, abstract: str, index: int = 0) -> Dict[str, Any]:
        """Extract from a single abstract with retries."""
        if pd.isna(abstract) or len(str(abstract)) < 50:
            return self._default()

        for attempt in range(MAX_RETRIES):
            result, raw = self._call_gemini(str(abstract))

            # Store debug info
            if DEBUG_MODE and len(self.debug_responses) < DEBUG_SAMPLE_SIZE:
                self.debug_responses.append({
                    "index": index,
                    "abstract": abstract[:200] + "...",
                    "raw_response": raw[:500] if isinstance(raw, str) else str(raw),
                    "parsed": result
                })

            if result.get("_retry"):
                time.sleep(2 * (attempt + 1))
                continue

            return result

        return self._default()

    def batch_extract_parallel(
        self,
        df: pd.DataFrame,
        abstract_col: str = 'Abstract',
        max_workers: int = 5,
        delay: float = 0.2
    ) -> pd.DataFrame:
        """Extract from all abstracts using parallel processing."""
        print("\n" + "=" * 60)
        print("🤖 EXTRACTING PAPER CLASSIFICATIONS WITH GEMINI")
        print("=" * 60)
        print(f"Model: {self.model}")
        print(f"Papers to process: {len(df)}")
        print(f"Parallel workers: {max_workers}")
        if DEBUG_MODE:
            print(f"🔍 DEBUG MODE ON - will show {DEBUG_SAMPLE_SIZE} raw responses")

        abstracts = df[abstract_col].tolist()
        results = [None] * len(abstracts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.extract_single, abstract, i): i
                for i, abstract in enumerate(abstracts)
            }

            for future in tqdm(as_completed(future_to_idx), total=len(abstracts), desc="Extracting"):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = self._default()
                time.sleep(delay)

        # Print debug info
        if DEBUG_MODE and self.debug_responses:
            print("\n" + "=" * 60)
            print("🔍 DEBUG: Sample Raw Responses")
            print("=" * 60)
            for d in self.debug_responses:
                print(f"\n--- Paper {d['index']} ---")
                print(f"Abstract: {d['abstract']}")
                print(f"Raw response: {d['raw_response']}")
                print(f"Parsed theory: {d['parsed']['theory']}")

        # Add results to DataFrame
        result_df = df.copy()
        for key in ['theory', 'paradigm', 'type', 'epistemic', 'anatomy', 'tone', 'target', 'subject', 'confidence']:
            result_df[key] = [r.get(key, self._default()[key]) for r in results]

        # Statistics
        print(f"\n✅ Extraction complete!")
        print(f"\n📊 Theory distribution:")
        for theory, count in result_df['theory'].value_counts().items():
            pct = count / len(result_df) * 100
            bar = "█" * int(pct / 2)
            print(f"   {theory:8s}: {count:5d} ({pct:5.1f}%) {bar}")

        # Quality check
        non_neutral = (result_df['theory'] != 'Neutral').sum()
        print(f"\n📈 Non-neutral classifications: {non_neutral} ({non_neutral/len(result_df)*100:.1f}%)")

        if non_neutral < len(result_df) * 0.05:
            print("\n⚠️  WARNING: Very few non-neutral classifications!")
            print("    Consider: 1) Check API responses with DEBUG_MODE=True")
            print("              2) The corpus may genuinely be mostly neutral")

        return result_df

# =============================================================================
# CELL 6: RULE-BASED EXTRACTOR (FALLBACK)
# =============================================================================

class RuleBasedExtractor:
    """Rule-based extraction using keyword matching - used as fallback."""

    THEORY_KEYWORDS = {
        'IIT': ['integrated information', 'phi', 'tononi', ' iit ', 'information integration',
                'exclusion postulate', 'intrinsic causal'],
        'GNWT': ['global workspace', 'global neuronal workspace', 'dehaene', 'baars',
                 'ignition', 'broadcasting', ' gnw ', 'access consciousness'],
        'HOT': ['higher-order', 'higher order thought', 'rosenthal', 'metacognition',
                'hot theory', 'higher-order representation'],
        'RPT': ['recurrent processing', 'lamme', ' rpt ', 'recurrent activity',
                'feedforward sweep', 'local recurrence']
    }

    ANATOMY_KEYWORDS = [
        'prefrontal', 'pfc', 'dlpfc', 'acc', 'anterior cingulate', 'parietal', 'ppc',
        'temporal', 'occipital', 'v1', 'v2', 'v4', 'visual cortex', 'thalamus',
        'claustrum', 'insula', 'tpj', 'temporo-parietal', 'mtl', 'hippocampus',
        'amygdala', 'basal ganglia', 'cerebellum', 'brainstem', 'posterior hot zone'
    ]

    def extract(self, abstract: str) -> Dict[str, Any]:
        """Extract using keyword matching."""
        if pd.isna(abstract) or not abstract:
            return self._default()

        text = ' ' + abstract.lower() + ' '

        # Theory
        scores = {t: sum(1 for kw in kws if kw in text) for t, kws in self.THEORY_KEYWORDS.items()}
        max_score = max(scores.values())
        theory = max(scores, key=scores.get) if max_score > 0 else "Neutral"

        # Anatomy
        anatomy = [kw.upper() for kw in self.ANATOMY_KEYWORDS if kw in text][:3]

        # Paradigm
        paradigm = "No-Report" if any(x in text for x in ['no-report', 'no report', 'unreportable']) else "Report"

        # Type
        type_val = "State" if any(x in text for x in ['disorder', 'coma', 'anesthesia', 'sleep', 'vegetative']) else "Content"

        # Target
        target = "Phenomenology" if any(x in text for x in ['qualia', 'phenomenal', 'subjective experience', 'what it is like']) else "Function"

        # Subject
        if any(x in text for x in ['patient', 'clinical', 'disorder']):
            subject = "Clinical"
        elif any(x in text for x in ['monkey', 'mouse', 'rat', 'animal', 'primate']):
            subject = "Animal"
        elif any(x in text for x in ['simulation', 'model', 'computational']):
            subject = "Simulation"
        elif any(x in text for x in ['review', 'meta-analysis']):
            subject = "Review"
        else:
            subject = "Human"

        return {
            "theory": theory, "paradigm": paradigm, "type": type_val,
            "epistemic": "Post-Hoc", "anatomy": anatomy, "tone": "Neutral",
            "target": target, "subject": subject, "confidence": min(max_score / 2, 1.0)
        }

    def _default(self) -> Dict[str, Any]:
        return {"theory": "Neutral", "paradigm": "Report", "type": "Content",
                "epistemic": "Post-Hoc", "anatomy": [], "tone": "Neutral",
                "target": "Function", "subject": "Human", "confidence": 0.0}

    def batch_extract(self, df: pd.DataFrame, abstract_col: str = 'Abstract') -> pd.DataFrame:
        """Extract from all abstracts."""
        print("\n" + "=" * 60)
        print("📝 EXTRACTING WITH RULE-BASED CLASSIFIER")
        print("=" * 60)

        results = [self.extract(row[abstract_col]) for _, row in tqdm(df.iterrows(), total=len(df))]

        result_df = df.copy()
        for key in ['theory', 'paradigm', 'type', 'epistemic', 'anatomy', 'tone', 'target', 'subject', 'confidence']:
            result_df[key] = [r[key] for r in results]

        print(f"\n📊 Theory distribution:")
        for t, c in result_df['theory'].value_counts().items():
            print(f"   {t}: {c} ({c/len(result_df)*100:.1f}%)")

        return result_df

# =============================================================================
# CELL 7: ANALYSIS 1 - CONTRAST AUDIT
# =============================================================================

def run_contrast_audit(df: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
    """Part 1: ConTraSt Audit - Does Methodology Predict Theory?"""
    print("\n" + "=" * 60)
    print("📊 ANALYSIS 1: CONTRAST AUDIT")
    print("Does Methodology Predict Theory?")
    print("=" * 60)

    df_theories = df[df['theory'] != 'Neutral'].copy()
    print(f"Papers with theory assignment: {len(df_theories)}")

    if len(df_theories) < 20:
        print("⚠️ Insufficient papers for analysis")
        return {"error": "Insufficient data", "n_papers": len(df_theories)}

    # Encode features
    le_paradigm = LabelEncoder()
    le_type = LabelEncoder()
    le_theory = LabelEncoder()

    df_theories['paradigm_enc'] = le_paradigm.fit_transform(df_theories['paradigm'])
    df_theories['type_enc'] = le_type.fit_transform(df_theories['type'])
    df_theories['theory_enc'] = le_theory.fit_transform(df_theories['theory'])

    X = df_theories[['paradigm_enc', 'type_enc']].values
    y = df_theories['theory_enc'].values

    # RandomForest
    print("\n🌲 Training RandomForest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    cv_scores = cross_val_score(rf, X, y, cv=min(5, len(set(y))))
    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

    rf.fit(X, y)
    print(f"Feature Importance - Paradigm: {rf.feature_importances_[0]:.3f}, Type: {rf.feature_importances_[1]:.3f}")

    # Sankey Diagram
    counts = df_theories.groupby(['paradigm', 'theory']).size().reset_index(name='count')
    paradigms = df_theories['paradigm'].unique().tolist()
    theories = df_theories['theory'].unique().tolist()

    sources, targets, values = [], [], []
    for _, row in counts.iterrows():
        sources.append(paradigms.index(row['paradigm']))
        targets.append(len(paradigms) + theories.index(row['theory']))
        values.append(row['count'])

    fig = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20, label=paradigms + theories,
                  color=['#2E86AB', '#A23B72'] + ['#F18F01', '#C73E1D', '#3B1F2B', '#95190C'][:len(theories)]),
        link=dict(source=sources, target=targets, value=values)
    ))

    fig.update_layout(
        title=f"<b>ConTraSt: Paradigm → Theory</b><br><sup>RF Accuracy: {cv_scores.mean():.1%} | n={len(df_theories)}</sup>",
        font=dict(size=12), height=500, width=800
    )

    fig.write_image(os.path.join(output_folder, "1_contrast_sankey.png"), scale=2)
    print(f"  ✅ Saved: 1_contrast_sankey.png")

    return {"cv_accuracy": cv_scores.mean(), "n_papers": len(df_theories)}

# =============================================================================
# CELL 8: ANALYSIS 2 - MOVING GOALPOSTS
# =============================================================================

def run_moving_goalposts(df: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
    """Part 2: Moving Goalposts - Semantic Drift Analysis"""
    print("\n" + "=" * 60)
    print("📈 ANALYSIS 2: MOVING GOALPOSTS")
    print("Semantic Drift Over Time")
    print("=" * 60)

    print("🤖 Loading sentence transformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    df_theories = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])].copy()
    print(f"Papers with theory: {len(df_theories)}")

    if len(df_theories) < 10:
        print("⚠️ Insufficient papers")
        return {"error": "Insufficient data"}

    df_theories['def_text'] = df_theories['Abstract'].apply(lambda x: extract_first_sentences(x, 2))

    drift_data = {}
    colors = {'IIT': '#E63946', 'GNWT': '#457B9D', 'HOT': '#2A9D8F', 'RPT': '#E9C46A'}

    for theory in ['IIT', 'GNWT', 'HOT', 'RPT']:
        tdf = df_theories[df_theories['theory'] == theory]
        if len(tdf) < 3:
            continue

        embeddings_by_year = {}
        for year in range(MIN_YEAR, MAX_YEAR + 1):
            ydf = tdf[tdf['Year'] == year]
            if len(ydf) >= 1:
                embeddings_by_year[year] = model.encode(ydf['def_text'].tolist())

        if not embeddings_by_year:
            continue

        years = sorted(embeddings_by_year.keys())
        base_year = BASELINE_YEAR if BASELINE_YEAR in years else years[0]
        baseline = np.mean(embeddings_by_year[base_year], axis=0)

        drift_data[theory] = [
            {"year": y, "drift": cosine(baseline, np.mean(embeddings_by_year[y], axis=0))}
            for y in years
        ]

    fig = go.Figure()
    for theory, data in drift_data.items():
        td = pd.DataFrame(data).sort_values('year')
        if len(td) > 3:
            td['drift'] = td['drift'].rolling(3, min_periods=1, center=True).mean()
        fig.add_trace(go.Scatter(
            x=td['year'], y=td['drift'], mode='lines+markers', name=theory,
            line=dict(color=colors[theory], width=3)
        ))

    fig.update_layout(
        title="<b>Moving Goalposts: Semantic Drift</b><br><sup>Cosine distance from baseline</sup>",
        xaxis_title="Year", yaxis_title="Semantic Drift",
        height=500, width=900
    )

    fig.write_image(os.path.join(output_folder, "2_moving_goalposts.png"), scale=2)
    print(f"  ✅ Saved: 2_moving_goalposts.png")

    return {"drift_data": drift_data}

# =============================================================================
# CELL 9: ANALYSIS 3 - ADVERSARIAL INDEX
# =============================================================================

def run_adversarial_index(df: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
    """Part 3: Adversarial Index - Cross-Theory Engagement"""
    print("\n" + "=" * 60)
    print("⚔️ ANALYSIS 3: ADVERSARIAL INDEX")
    print("Cross-Theory Citations")
    print("=" * 60)

    patterns = {
        'IIT': r'\b(integrated information|iit\b|tononi)',
        'GNWT': r'\b(global.?workspace|gnw|dehaene|baars)',
        'HOT': r'\b(higher.?order|hot theory|rosenthal)',
        'RPT': r'\b(recurrent processing|rpt\b|lamme)'
    }

    def count_theories(text):
        if pd.isna(text): return []
        return [t for t, p in patterns.items() if re.search(p, text.lower())]

    df['theories_mentioned'] = df['Abstract'].apply(count_theories)
    df['is_adversarial'] = df['theories_mentioned'].apply(len) >= 2

    yearly = df.groupby('Year').agg(
        total=('Title', 'count'),
        adversarial=('is_adversarial', 'sum')
    ).reset_index()
    yearly['pct'] = yearly['adversarial'] / yearly['total'] * 100
    yearly = yearly[(yearly['Year'] >= MIN_YEAR) & (yearly['Year'] <= MAX_YEAR)]

    print(f"Adversarial papers: {df['is_adversarial'].sum()} ({df['is_adversarial'].mean()*100:.1f}%)")

    z = np.polyfit(yearly['Year'], yearly['pct'], 1)
    trend = "increasing" if z[0] > 0 else "decreasing"

    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly['Year'], y=yearly['pct'], marker_color='#E63946'))
    fig.add_trace(go.Scatter(x=yearly['Year'], y=np.poly1d(z)(yearly['Year']),
                             mode='lines', line=dict(color='#1D3557', dash='dash'), name='Trend'))

    fig.update_layout(
        title=f"<b>Adversarial Index</b><br><sup>Papers citing 2+ theories | Trend: {trend}</sup>",
        xaxis_title="Year", yaxis_title="% of Papers",
        height=500, width=900
    )

    fig.write_image(os.path.join(output_folder, "3_adversarial_index.png"), scale=2)
    print(f"  ✅ Saved: 3_adversarial_index.png")

    return {"total": int(df['is_adversarial'].sum()), "trend": trend}

# =============================================================================
# CELL 10: ANALYSIS 4 - NEURO-CARTOGRAPHY
# =============================================================================

def run_neuro_cartography(df: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
    """Part 4: Neuro-Cartography - Brain Region Biases"""
    print("\n" + "=" * 60)
    print("🧠 ANALYSIS 4: NEURO-CARTOGRAPHY")
    print("Brain Region Frequency by Theory")
    print("=" * 60)

    df_theories = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])].copy()

    if len(df_theories) < 10:
        print("⚠️ Insufficient papers")
        return {"error": "Insufficient data"}

    # Collect all regions
    region_counts = defaultdict(lambda: defaultdict(int))

    for _, row in df_theories.iterrows():
        theory = row['theory']
        anatomy = row.get('anatomy', [])
        if isinstance(anatomy, list):
            for region in anatomy:
                if region and str(region).lower() != 'none':
                    region_counts[theory][str(region).upper()] += 1

    # Get top regions
    all_regions = Counter()
    for theory_regions in region_counts.values():
        all_regions.update(theory_regions)

    top_regions = [r for r, _ in all_regions.most_common(15)]

    if not top_regions:
        print("⚠️ No brain regions extracted")
        return {"error": "No regions found"}

    # Build matrix
    theories = ['IIT', 'GNWT', 'HOT', 'RPT']
    matrix = []
    for theory in theories:
        row = [region_counts[theory].get(r, 0) for r in top_regions]
        # Normalize by total papers for theory
        total = sum(row) if sum(row) > 0 else 1
        matrix.append([x / total * 100 for x in row])

    fig = go.Figure(go.Heatmap(
        z=matrix, x=top_regions, y=theories,
        colorscale='Reds', text=[[f"{v:.0f}%" for v in row] for row in matrix],
        texttemplate="%{text}", textfont={"size": 10}
    ))

    fig.update_layout(
        title="<b>Neuro-Cartography: Brain Regions by Theory</b><br><sup>Normalized frequency (%)</sup>",
        height=400, width=1000
    )

    fig.write_image(os.path.join(output_folder, "4_neuro_cartography.png"), scale=2)
    print(f"  ✅ Saved: 4_neuro_cartography.png")

    return {"top_regions": top_regions}

# =============================================================================
# CELL 11: ANALYSIS 5 - AGGRESSION INDEX
# =============================================================================

def run_aggression_index(df: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
    """Part 5: Aggression Index - Hostility Over Time"""
    print("\n" + "=" * 60)
    print("😤 ANALYSIS 5: AGGRESSION INDEX")
    print("Hostility Trends Over Time")
    print("=" * 60)

    df_theories = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])].copy()

    # Count hostile tones by year
    yearly = df_theories.groupby('Year').apply(
        lambda x: pd.Series({
            'total': len(x),
            'dismissive': (x['tone'] == 'Dismissive').sum(),
            'critical': (x['tone'] == 'Critical').sum(),
            'constructive': (x['tone'] == 'Constructive').sum()
        })
    ).reset_index()

    yearly['hostility_pct'] = (yearly['dismissive'] + yearly['critical']) / yearly['total'] * 100
    yearly = yearly[(yearly['Year'] >= MIN_YEAR) & (yearly['Year'] <= MAX_YEAR)]

    if len(yearly) < 3:
        print("⚠️ Insufficient data")
        return {"error": "Insufficient data"}

    z = np.polyfit(yearly['Year'], yearly['hostility_pct'], 1)
    trend = "increasing" if z[0] > 0 else "decreasing"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly['Year'], y=yearly['hostility_pct'], mode='lines+markers',
        line=dict(color='#E63946', width=3), name='Hostility %'
    ))
    fig.add_trace(go.Scatter(
        x=yearly['Year'], y=np.poly1d(z)(yearly['Year']),
        mode='lines', line=dict(color='#1D3557', dash='dash'), name='Trend'
    ))

    fig.update_layout(
        title=f"<b>Aggression Index: Hostility Over Time</b><br><sup>% Dismissive + Critical | Trend: {trend}</sup>",
        xaxis_title="Year", yaxis_title="Hostility %",
        height=500, width=900
    )

    fig.write_image(os.path.join(output_folder, "5_aggression_index.png"), scale=2)
    print(f"  ✅ Saved: 5_aggression_index.png")

    return {"trend": trend}

# =============================================================================
# CELL 12: ANALYSIS 6 - EASY VS HARD PROBLEM
# =============================================================================

def run_easy_vs_hard(df: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
    """Part 6: Easy vs Hard Problem Analysis"""
    print("\n" + "=" * 60)
    print("🤔 ANALYSIS 6: EASY VS HARD PROBLEM")
    print("Phenomenology vs Function")
    print("=" * 60)

    df_theories = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])].copy()

    if len(df_theories) < 10:
        print("⚠️ Insufficient papers")
        return {"error": "Insufficient data"}

    # Count targets by theory
    counts = df_theories.groupby(['theory', 'target']).size().unstack(fill_value=0)

    # Ensure all columns exist
    for col in ['Phenomenology', 'Function', 'Mechanism']:
        if col not in counts.columns:
            counts[col] = 0

    # Normalize
    counts_pct = counts.div(counts.sum(axis=1), axis=0) * 100

    fig = go.Figure()
    colors = {'Phenomenology': '#E63946', 'Function': '#457B9D', 'Mechanism': '#2A9D8F'}

    for target in ['Phenomenology', 'Function', 'Mechanism']:
        if target in counts_pct.columns:
            fig.add_trace(go.Bar(
                x=counts_pct.index, y=counts_pct[target],
                name=target, marker_color=colors[target]
            ))

    fig.update_layout(
        title="<b>Easy vs Hard Problem</b><br><sup>What does each theory explain?</sup>",
        xaxis_title="Theory", yaxis_title="% of Papers",
        barmode='stack', height=500, width=800
    )

    fig.write_image(os.path.join(output_folder, "6_easy_vs_hard.png"), scale=2)
    print(f"  ✅ Saved: 6_easy_vs_hard.png")

    return {"counts": counts.to_dict()}

# =============================================================================
# CELL 13: ANALYSIS 7 - MODEL ORGANISM BIAS
# =============================================================================

def run_model_organism(df: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
    """Part 7: Model Organism Bias - Subject Distribution"""
    print("\n" + "=" * 60)
    print("🐁 ANALYSIS 7: MODEL ORGANISM BIAS")
    print("Subject Distribution by Theory")
    print("=" * 60)

    df_theories = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])].copy()

    if len(df_theories) < 10:
        print("⚠️ Insufficient papers")
        return {"error": "Insufficient data"}

    # Count subjects by theory
    counts = df_theories.groupby(['theory', 'subject']).size().unstack(fill_value=0)

    # Normalize
    counts_pct = counts.div(counts.sum(axis=1), axis=0) * 100

    fig = go.Figure()
    colors = {'Human': '#2E86AB', 'Clinical': '#A23B72', 'Animal': '#F18F01',
              'Simulation': '#C73E1D', 'Review': '#6C757D'}

    for subject in ['Human', 'Clinical', 'Animal', 'Simulation', 'Review']:
        if subject in counts_pct.columns:
            fig.add_trace(go.Bar(
                x=counts_pct.index, y=counts_pct[subject],
                name=subject, marker_color=colors.get(subject, '#333')
            ))

    fig.update_layout(
        title="<b>Model Organism Bias</b><br><sup>What data does each theory use?</sup>",
        xaxis_title="Theory", yaxis_title="% of Papers",
        barmode='stack', height=500, width=800
    )

    fig.write_image(os.path.join(output_folder, "7_model_organism.png"), scale=2)
    print(f"  ✅ Saved: 7_model_organism.png")

    return {"counts": counts.to_dict()}

# =============================================================================
# CELL 14: MAIN PIPELINE
# =============================================================================

def run_full_audit(
    csv_path: str = CSV_PATH,
    api_key: str = GEMINI_API_KEY,
    output_folder: str = OUTPUT_FOLDER,
    use_llm: bool = True,
    use_rules_only: bool = False
) -> Dict[str, Any]:
    """Run the complete 7-analysis audit."""

    print("\n" + "=" * 70)
    print("    🧠 CONSCIOUSNESS WARS BIBLIOMETRIC AUDIT v2.0 🧠")
    print("    IIT vs GNWT vs HOT vs RPT")
    print("    8 Dimensions × 7 Analyses")
    print("=" * 70)

    os.makedirs(output_folder, exist_ok=True)

    # Load data
    df = load_bibliography(csv_path)

    # Extract classifications
    if use_rules_only:
        print("\n📝 Using RULE-BASED extraction only")
        df_classified = RuleBasedExtractor().batch_extract(df)
    elif use_llm and api_key:
        try:
            extractor = GeminiExtractor(api_key, GEMINI_MODEL)
            test = extractor.extract_single("This paper tests integrated information theory using phi measures.", 0)
            print(f"✅ Gemini connected | Test: {test['theory']}")
            df_classified = extractor.batch_extract_parallel(df, max_workers=MAX_WORKERS, delay=API_DELAY)
        except Exception as e:
            print(f"⚠️ Gemini failed: {e}")
            print("Falling back to rule-based...")
            df_classified = RuleBasedExtractor().batch_extract(df)
    else:
        df_classified = RuleBasedExtractor().batch_extract(df)

    # Run all analyses
    results = {}

    print("\n" + "=" * 60)
    print("📊 RUNNING 7 ANALYSES")
    print("=" * 60)

    try:
        results['contrast'] = run_contrast_audit(df_classified, output_folder)
    except Exception as e:
        print(f"⚠️ Analysis 1 failed: {e}")

    try:
        results['goalposts'] = run_moving_goalposts(df_classified, output_folder)
    except Exception as e:
        print(f"⚠️ Analysis 2 failed: {e}")

    try:
        results['adversarial'] = run_adversarial_index(df_classified, output_folder)
    except Exception as e:
        print(f"⚠️ Analysis 3 failed: {e}")

    try:
        results['cartography'] = run_neuro_cartography(df_classified, output_folder)
    except Exception as e:
        print(f"⚠️ Analysis 4 failed: {e}")

    try:
        results['aggression'] = run_aggression_index(df_classified, output_folder)
    except Exception as e:
        print(f"⚠️ Analysis 5 failed: {e}")

    try:
        results['easy_hard'] = run_easy_vs_hard(df_classified, output_folder)
    except Exception as e:
        print(f"⚠️ Analysis 6 failed: {e}")

    try:
        results['organism'] = run_model_organism(df_classified, output_folder)
    except Exception as e:
        print(f"⚠️ Analysis 7 failed: {e}")

    # Save data
    df_classified.to_csv(os.path.join(output_folder, "classified_papers.csv"), index=False)
    print(f"\n💾 Saved: classified_papers.csv")

    with open(os.path.join(output_folder, "results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"💾 Saved: results.json")

    # Summary
    print("\n" + "=" * 70)
    print("    ✅ AUDIT COMPLETE")
    print("=" * 70)
    print(f"\n📊 Papers analyzed: {len(df_classified)}")
    print(f"\n📈 Theory distribution:")
    for t, c in df_classified['theory'].value_counts().items():
        print(f"   {t}: {c} ({c/len(df_classified)*100:.1f}%)")

    print(f"\n📁 Results saved to: {output_folder}")

    return {'data': df_classified, 'results': results}


# =============================================================================
# CELL 15: RUN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CONSCIOUSNESS WARS AUDIT")
    print("=" * 60)
    print("\nTo run:")
    print("  1. Set GEMINI_API_KEY at the top")
    print("  2. Call: results = run_full_audit()")
    print("\nOptions:")
    print("  - use_llm=True  → Use Gemini (default)")
    print("  - use_rules_only=True → Skip Gemini, use keywords only")
    print("  - DEBUG_MODE=True → See raw API responses")
