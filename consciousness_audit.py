"""
================================================================================
CONSCIOUSNESS WARS BIBLIOMETRIC AUDIT PIPELINE v5.1
================================================================================
A comprehensive bibliometric analysis of consciousness theories:
IIT (Integrated Information Theory), GNWT (Global Neuronal Workspace Theory),
HOT (Higher-Order Theory), and RPT (Recurrent Processing Theory).

v5.0 UPGRADES:
- Multi-Persona Debate Classifier (Proponent → Skeptic → Judge workflow)
- BERTopic Dynamic Topic Modeling with topics_over_time
- Argumentation Mining (Claims, Premises, Support/Attack relations)

ANALYSES:
1-7: Original analyses (ConTraSt, Moving Goalposts, Adversarial, etc.)
8: Semantic Space Clustering with UMAP visualization
9: Hypothesis Testing (Implicit Bias, Schism, Insularity)
10: Methodology Predicts Theory (Yaron et al. 2022)
11: Vocabulary Divergence (Echo chamber analysis)
12: Dogmatism Analysis
13: BERTopic Topic Modeling (NEW)
14: Argumentation Mining (NEW)

================================================================================
"""

# =============================================================================
# CELL 1: CONFIGURATION
# =============================================================================

import os

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

_env = load_env_file()
GEMINI_API_KEY = _env.get('GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY', ''))

OUTPUT_FOLDER = "./results"
CSV_PATH = "cons_bib.csv"

# Model - supports gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro
GEMINI_MODEL = _env.get('GEMINI_MODEL', 'gemini-2.0-flash')
EMBEDDING_MODEL = "text-embedding-004"

MAX_WORKERS = 8
API_DELAY = 0.2
MAX_RETRIES = 3
EMBEDDING_BATCH_SIZE = 100

BASELINE_YEAR = 2005
MIN_YEAR = 2000
MAX_YEAR = 2025
SCHISM_YEAR = 2015

DEBUG_MODE = True
DEBUG_SAMPLE_SIZE = 5

# Enable/disable v5.0 features (can be slow)
ENABLE_DEBATE_CLASSIFIER = True  # Multi-persona debate
ENABLE_BERTOPIC = True           # BERTopic analysis
ENABLE_ARGUMENT_MINING = True    # Argumentation mining

# =============================================================================
# CELL 2: IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import json
import re
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cosine, euclidean
from scipy import stats
import requests
from tqdm import tqdm

# Optional imports
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("⚠️ sentence-transformers not available")

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

try:
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from sklearn.feature_extraction.text import CountVectorizer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("⚠️ BERTopic not available. Install with: pip install bertopic")

warnings.filterwarnings('ignore')

# =============================================================================
# CELL 3: DATA LOADING
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

# =============================================================================
# CELL 4: PROMPTS FOR MULTI-PERSONA DEBATE CLASSIFIER
# =============================================================================

PROPONENT_PROMPT = '''You are Agent A: THE PROPONENT. Your job is to find evidence SUPPORTING a consciousness theory.

ABSTRACT:
{abstract}

THEORIES TO CONSIDER:
- IIT (Integrated Information Theory): Phi, integrated information, Tononi, posterior hot zone
- GNWT (Global Workspace Theory): Global workspace, ignition, Dehaene, broadcasting, P300
- HOT (Higher-Order Theory): Higher-order thought, metacognition, Rosenthal, Lau
- RPT (Recurrent Processing Theory): Recurrent processing, Lamme, feedforward/feedback in visual cortex

Analyze the abstract and identify:
1. Which theory (if any) does this abstract SUPPORT?
2. What specific evidence/keywords support this theory?
3. How strong is the support (weak/moderate/strong)?

Return JSON only:
{{"supported_theory": "IIT|GNWT|HOT|RPT|None", "evidence": ["list", "of", "evidence"], "strength": "weak|moderate|strong", "reasoning": "brief explanation"}}'''

SKEPTIC_PROMPT = '''You are Agent B: THE SKEPTIC. Your job is to CHALLENGE the proponent's analysis and find counter-evidence.

ABSTRACT:
{abstract}

PROPONENT'S CLAIM: The abstract supports {claimed_theory} with {strength} evidence.
PROPONENT'S EVIDENCE: {evidence}

Your task:
1. Find evidence that CONTRADICTS this classification
2. Identify if another theory might be better supported
3. Note any methodological issues or ambiguity

Return JSON only:
{{"challenges": ["list", "of", "counter-arguments"], "alternative_theory": "IIT|GNWT|HOT|RPT|None", "alternative_evidence": ["evidence", "for", "alternative"], "confidence_reduction": 0.0-0.5}}'''

JUDGE_PROMPT = '''You are Agent C: THE JUDGE. Make the final classification based on the debate.

ABSTRACT:
{abstract}

PROPONENT (Agent A) argues for: {proponent_theory}
- Evidence: {proponent_evidence}
- Strength: {proponent_strength}

SKEPTIC (Agent B) challenges:
- Counter-arguments: {skeptic_challenges}
- Alternative theory: {skeptic_alternative}
- Alternative evidence: {skeptic_evidence}

Make your FINAL RULING:
1. Which theory is best supported? (IIT, GNWT, HOT, RPT, or Neutral)
2. Final confidence score (0.0-1.0)
3. Additional dimensions to classify

Return JSON only:
{{"theory": "IIT|GNWT|HOT|RPT|Neutral", "confidence": 0.0-1.0, "paradigm": "Report|No-Report", "type": "Content|State", "epistemic": "A Priori|Post-Hoc", "anatomy": ["region1"], "tone": "Dismissive|Critical|Constructive|Neutral", "target": "Phenomenology|Function|Mechanism", "subject": "Human|Clinical|Animal|Simulation|Review", "ruling_reasoning": "brief explanation"}}'''

# Simple fallback prompt
SIMPLE_CLASSIFICATION_PROMPT = '''Classify this neuroscience abstract.

ABSTRACT:
{abstract}

THEORIES:
- IIT: "integrated information", "phi", "Tononi", "posterior hot zone"
- GNWT: "global workspace", "ignition", "Dehaene", "broadcasting"
- HOT: "higher-order", "metacognition", "Rosenthal"
- RPT: "recurrent processing", "Lamme", "feedforward"
- Neutral: No clear theory

Return JSON:
{{"theory": "IIT|GNWT|HOT|RPT|Neutral", "confidence": 0.0-1.0, "paradigm": "Report|No-Report", "type": "Content|State", "tone": "Neutral|Critical|Dismissive|Constructive", "target": "Function|Mechanism|Phenomenology", "subject": "Human|Clinical|Animal|Simulation|Review"}}'''

# =============================================================================
# CELL 5: ARGUMENTATION MINING PROMPT
# =============================================================================

ARGUMENT_MINING_PROMPT = '''Extract the argumentative structure from this abstract.

ABSTRACT:
{abstract}

Identify:
1. MAIN CLAIM: What is the paper's central thesis/finding?
2. PREMISES: What evidence/reasoning supports this claim?
3. THEORETICAL TARGETS: Which theories does this paper support or attack?

For theoretical targets, use format: [SUPPORTS|ATTACKS] [THEORY] because [REASON]

Return JSON:
{{"main_claim": "the central thesis", "premises": ["premise 1", "premise 2"], "theoretical_relations": [{{"relation": "SUPPORTS|ATTACKS", "theory": "IIT|GNWT|HOT|RPT|consciousness_general", "reason": "why"}}], "argument_strength": "weak|moderate|strong"}}'''

# =============================================================================
# CELL 6: GEMINI EXTRACTOR v5.0 (Multi-Persona Debate)
# =============================================================================

class GeminiExtractor:
    """v5.0 Extractor with Multi-Persona Debate workflow."""

    def __init__(self, api_key: str, model: str = GEMINI_MODEL):
        self.api_key = api_key
        self.model = model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.session = requests.Session()
        self.debug_responses = []
        self.debate_logs = []
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not set!")

    def _call_api(self, prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> Tuple[Optional[str], str]:
        """Make API call and return raw text response."""
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        try:
            r = self.session.post(
                f"{self.base_url}?key={self.api_key}",
                json=payload,
                timeout=60
            )
            if r.status_code == 429:
                time.sleep(10)
                return None, "RATE_LIMITED"
            if r.status_code != 200:
                if DEBUG_MODE:
                    print(f"  ⚠️ API Error {r.status_code}: {r.text[:150]}")
                return None, f"HTTP_{r.status_code}"

            result = r.json()
            if 'candidates' in result and result['candidates']:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    text = candidate['content']['parts'][0].get('text', '')
                    return text, "OK"
                if candidate.get('finishReason') == 'SAFETY':
                    return None, "SAFETY_BLOCKED"
            return None, "NO_CANDIDATES"
        except requests.exceptions.Timeout:
            return None, "TIMEOUT"
        except Exception as e:
            return None, f"ERROR_{str(e)[:30]}"

    def _parse_json(self, text: str) -> Optional[Dict]:
        """Parse JSON from response text with robust extraction."""
        if not text:
            return None

        original_text = text

        # Clean markdown code blocks
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
        text = text.strip()

        # Direct parse
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except:
            pass

        # Find balanced braces
        try:
            start = text.find('{')
            if start >= 0:
                depth = 0
                for i, c in enumerate(text[start:], start):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            json_str = text[start:i+1]
                            result = json.loads(json_str)
                            if isinstance(result, dict):
                                return result
                            break
        except:
            pass

        # Try fixing common JSON issues
        try:
            # Find JSON portion and clean it
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                # Fix single quotes to double quotes
                json_str = re.sub(r"'([^']+)':", r'"\1":', json_str)
                json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
                # Fix trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                result = json.loads(json_str)
                if isinstance(result, dict):
                    return result
        except:
            pass

        # Last resort: extract key-value pairs manually
        try:
            result = {}
            # Extract theory
            theory_match = re.search(r'"?theory"?\s*[:\s]+["\']?(IIT|GNWT|HOT|RPT|Neutral)["\']?', text, re.I)
            if theory_match:
                result['theory'] = theory_match.group(1).upper()
            # Extract confidence
            conf_match = re.search(r'"?confidence"?\s*[:\s]+([0-9.]+)', text, re.I)
            if conf_match:
                result['confidence'] = float(conf_match.group(1))
            # Extract paradigm
            para_match = re.search(r'"?paradigm"?\s*[:\s]+["\']?(Report|No-Report)["\']?', text, re.I)
            if para_match:
                result['paradigm'] = para_match.group(1)
            # Extract tone
            tone_match = re.search(r'"?tone"?\s*[:\s]+["\']?(Neutral|Critical|Dismissive|Constructive)["\']?', text, re.I)
            if tone_match:
                result['tone'] = tone_match.group(1)

            if result and 'theory' in result:
                if DEBUG_MODE and len(self.debug_responses) < 3:
                    print(f"    [DEBUG] Regex extraction: {result}")
                return result
        except:
            pass

        if DEBUG_MODE:
            print(f"    [DEBUG] JSON parse failed. Raw text: {original_text[:200]}...")

        return None

    def _default_classification(self) -> Dict:
        return {
            "theory": "Neutral", "paradigm": "Report", "type": "Content",
            "epistemic": "Post-Hoc", "anatomy": [], "tone": "Neutral",
            "target": "Function", "subject": "Human", "confidence": 0.0,
            "fingerprints": [], "dogmatism_score": 5, "confidence_markers": [],
            "hedging_markers": [], "_extraction_failed": True,
            "debate_used": False, "proponent_theory": None, "skeptic_alternative": None
        }

    def _normalize(self, d: Dict, debate_used: bool = False) -> Dict:
        """Normalize parsed JSON to standard format."""
        r = self._default_classification()
        r["_extraction_failed"] = False
        r["debate_used"] = debate_used

        # Theory
        theory = str(d.get("theory", "")).strip().upper()
        if theory in ["IIT", "GNWT", "HOT", "RPT"]:
            r["theory"] = theory
        elif "IIT" in theory or "INTEGRATED" in theory:
            r["theory"] = "IIT"
        elif "GNWT" in theory or "GNW" in theory or "WORKSPACE" in theory:
            r["theory"] = "GNWT"
        elif "HOT" in theory or "HIGHER" in theory:
            r["theory"] = "HOT"
        elif "RPT" in theory or "RECURRENT" in theory:
            r["theory"] = "RPT"
        else:
            r["theory"] = "Neutral"

        # Confidence
        try:
            r["confidence"] = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
        except:
            r["confidence"] = 0.5

        # Paradigm
        paradigm = str(d.get("paradigm", "")).lower()
        r["paradigm"] = "No-Report" if "no" in paradigm else "Report"

        # Type
        type_val = str(d.get("type", "")).lower()
        r["type"] = "State" if "state" in type_val else "Content"

        # Epistemic
        epistemic = str(d.get("epistemic", "")).lower()
        r["epistemic"] = "A Priori" if "priori" in epistemic else "Post-Hoc"

        # Anatomy
        anat = d.get("anatomy", [])
        if isinstance(anat, str):
            anat = [a.strip() for a in anat.split(",")]
        if isinstance(anat, list):
            r["anatomy"] = [str(a).strip().upper() for a in anat
                          if a and str(a).lower() not in ["none", ""]][:3]

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
        if "phenom" in target:
            r["target"] = "Phenomenology"
        elif "mechan" in target:
            r["target"] = "Mechanism"
        else:
            r["target"] = "Function"

        # Subject
        subject = str(d.get("subject", "")).lower()
        if "clinical" in subject or "patient" in subject:
            r["subject"] = "Clinical"
        elif "animal" in subject:
            r["subject"] = "Animal"
        elif "simul" in subject or "model" in subject:
            r["subject"] = "Simulation"
        elif "review" in subject:
            r["subject"] = "Review"
        else:
            r["subject"] = "Human"

        return r

    def extract_with_debate(self, abstract: str) -> Dict:
        """
        Multi-Persona Debate Classification (v5.0 feature).
        Uses 3 agents: Proponent → Skeptic → Judge
        """
        if pd.isna(abstract) or len(str(abstract)) < 50:
            return self._default_classification()

        abstract = str(abstract)[:2500]

        # STEP 1: Proponent analysis
        proponent_prompt = PROPONENT_PROMPT.format(abstract=abstract)
        proponent_text, status = self._call_api(proponent_prompt, temperature=0.3)

        if status != "OK" or not proponent_text:
            # Fallback to simple classification
            return self.extract_simple(abstract)

        proponent_data = self._parse_json(proponent_text)
        if not proponent_data:
            return self.extract_simple(abstract)

        claimed_theory = proponent_data.get("supported_theory", "None")
        evidence = proponent_data.get("evidence", [])
        strength = proponent_data.get("strength", "weak")

        # If proponent found nothing, use simple classification
        if claimed_theory == "None" or not evidence:
            return self.extract_simple(abstract)

        # STEP 2: Skeptic challenges
        skeptic_prompt = SKEPTIC_PROMPT.format(
            abstract=abstract,
            claimed_theory=claimed_theory,
            strength=strength,
            evidence=", ".join(evidence) if isinstance(evidence, list) else str(evidence)
        )
        skeptic_text, status = self._call_api(skeptic_prompt, temperature=0.3)

        skeptic_data = {}
        if status == "OK" and skeptic_text:
            skeptic_data = self._parse_json(skeptic_text) or {}

        # STEP 3: Judge makes final ruling
        judge_prompt = JUDGE_PROMPT.format(
            abstract=abstract,
            proponent_theory=claimed_theory,
            proponent_evidence=", ".join(evidence) if isinstance(evidence, list) else str(evidence),
            proponent_strength=strength,
            skeptic_challenges=", ".join(skeptic_data.get("challenges", ["none"])),
            skeptic_alternative=skeptic_data.get("alternative_theory", "None"),
            skeptic_evidence=", ".join(skeptic_data.get("alternative_evidence", ["none"]))
        )
        judge_text, status = self._call_api(judge_prompt, temperature=0.1)

        if status != "OK" or not judge_text:
            # Use proponent's answer with reduced confidence
            result = self._default_classification()
            result["theory"] = claimed_theory if claimed_theory in ["IIT", "GNWT", "HOT", "RPT"] else "Neutral"
            result["confidence"] = 0.5
            result["_extraction_failed"] = False
            result["debate_used"] = True
            return result

        judge_data = self._parse_json(judge_text)
        if not judge_data:
            result = self._default_classification()
            result["theory"] = claimed_theory if claimed_theory in ["IIT", "GNWT", "HOT", "RPT"] else "Neutral"
            result["confidence"] = 0.5
            result["_extraction_failed"] = False
            result["debate_used"] = True
            return result

        # Normalize judge's ruling
        result = self._normalize(judge_data, debate_used=True)
        result["proponent_theory"] = claimed_theory
        result["skeptic_alternative"] = skeptic_data.get("alternative_theory")

        # Log debate for analysis
        if len(self.debate_logs) < 20:
            self.debate_logs.append({
                "proponent": proponent_data,
                "skeptic": skeptic_data,
                "judge": judge_data,
                "final": result["theory"]
            })

        return result

    def extract_simple(self, abstract: str) -> Dict:
        """Simple single-shot classification (fallback)."""
        if pd.isna(abstract) or len(str(abstract)) < 50:
            return self._default_classification()

        prompt = SIMPLE_CLASSIFICATION_PROMPT.format(abstract=str(abstract)[:2500])

        for attempt in range(MAX_RETRIES):
            text, status = self._call_api(prompt, temperature=0.1)

            if status == "RATE_LIMITED":
                time.sleep(5 * (attempt + 1))
                continue

            if status == "OK" and text:
                parsed = self._parse_json(text)
                if parsed:
                    result = self._normalize(parsed)
                    if DEBUG_MODE and len(self.debug_responses) < DEBUG_SAMPLE_SIZE:
                        self.debug_responses.append({
                            "raw": text[:200], "parsed": parsed, "normalized": result
                        })
                    return result

            time.sleep(1)

        return self._default_classification()

    def extract_single(self, abstract: str, idx: int = 0, use_debate: bool = False) -> Dict:
        """Extract classification for a single abstract."""
        try:
            if use_debate and ENABLE_DEBATE_CLASSIFIER:
                result = self.extract_with_debate(abstract)
            else:
                result = self.extract_simple(abstract)

            # Ensure all required keys exist
            result.setdefault("dogmatism_score", 5)
            result.setdefault("confidence_markers", [])
            result.setdefault("hedging_markers", [])
            return result

        except Exception as e:
            if DEBUG_MODE:
                print(f"  ⚠️ Extract error at {idx}: {type(e).__name__}: {e}")
            return self._default_classification()

    def batch_extract_parallel(self, df: pd.DataFrame, max_workers: int = MAX_WORKERS,
                               use_debate: bool = False) -> pd.DataFrame:
        """Extract dimensions from DataFrame in parallel."""
        mode = "DEBATE MODE (3 agents)" if use_debate else "SIMPLE MODE"
        print(f"\n{'='*60}\n🔍 EXTRACTING DIMENSIONS WITH GEMINI ({mode})\n{'='*60}")
        print(f"Model: {self.model} | Papers: {len(df)} | Workers: {max_workers}")

        if use_debate:
            print("⚡ Using Multi-Persona Debate: Proponent → Skeptic → Judge")

        # Test extraction with verbose output
        print("\n🧪 Testing extraction...")
        test_abstract = df.iloc[0]['Abstract']
        print(f"   Test abstract: {str(test_abstract)[:100]}...")

        # Direct API test to see raw response
        test_prompt = SIMPLE_CLASSIFICATION_PROMPT.format(abstract=str(test_abstract)[:1500])
        raw_text, status = self._call_api(test_prompt, temperature=0.1)
        print(f"   API status: {status}")
        if raw_text:
            print(f"   Raw response: {raw_text[:300]}...")
            parsed = self._parse_json(raw_text)
            print(f"   Parsed: {parsed}")

        test_result = self.extract_single(test_abstract, 0, use_debate=False)
        print(f"   Test result: theory={test_result.get('theory', 'N/A')}, conf={test_result.get('confidence', 0):.2f}")

        if test_result.get('_extraction_failed', True):
            print("   ⚠️ Test failed - JSON parsing issue")
            print("   Continuing with rule-based fallback for failed extractions...")
        else:
            print("   ✅ Test succeeded!")

        results = [None] * len(df)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.extract_single, row['Abstract'], i, use_debate): i
                for i, row in df.iterrows()
            }

            for future in tqdm(as_completed(futures), total=len(df), desc="Extracting"):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"⚠️ Error at {idx}: {e}")
                    results[idx] = self._default_classification()
                time.sleep(API_DELAY)

        # Apply to dataframe
        df_out = df.copy()
        keys = ['theory', 'paradigm', 'type', 'epistemic', 'anatomy', 'tone', 'target',
                'subject', 'confidence', 'fingerprints', 'dogmatism_score',
                '_extraction_failed', 'debate_used']

        for k in keys:
            default_val = self._default_classification().get(k)
            df_out[k] = [r.get(k, default_val) if r else default_val for r in results]

        # Stats
        print(f"\n📊 Extraction Statistics:")
        failed = df_out['_extraction_failed'].sum()
        debate_used = df_out['debate_used'].sum() if 'debate_used' in df_out.columns else 0
        print(f"   Success rate: {(1 - failed/len(df_out))*100:.1f}%")
        print(f"   Debate mode used: {debate_used} papers")
        print(f"   Mean confidence: {df_out['confidence'].mean():.3f}")

        print(f"\n📊 Theory Distribution:")
        for t, c in df_out['theory'].value_counts().items():
            print(f"   {t}: {c} ({c/len(df_out)*100:.1f}%)")

        return df_out

    def extract_arguments(self, abstract: str) -> Dict:
        """Extract argumentative structure from abstract."""
        default = {"main_claim": "", "premises": [], "theoretical_relations": [], "argument_strength": "weak"}

        if pd.isna(abstract) or len(str(abstract)) < 50:
            return default

        prompt = ARGUMENT_MINING_PROMPT.format(abstract=str(abstract)[:2500])
        text, status = self._call_api(prompt, temperature=0.2, max_tokens=800)

        if status == "OK" and text:
            parsed = self._parse_json(text)
            if parsed:
                result = {
                    "main_claim": parsed.get("main_claim", ""),
                    "premises": parsed.get("premises", []),
                    "theoretical_relations": parsed.get("theoretical_relations", []),
                    "argument_strength": parsed.get("argument_strength", "weak")
                }
                return result

            # Fallback: try to extract main_claim with regex
            claim_match = re.search(r'"?main_claim"?\s*[:\s]+["\']?([^"\'{}]+)["\']?', text, re.I)
            if claim_match:
                result = default.copy()
                result["main_claim"] = claim_match.group(1).strip()[:200]

                # Try to extract relations
                relations = []
                for match in re.finditer(r'(SUPPORTS?|ATTACKS?)\s+(IIT|GNWT|HOT|RPT)', text, re.I):
                    relations.append({
                        "relation": "SUPPORTS" if "SUPPORT" in match.group(1).upper() else "ATTACKS",
                        "theory": match.group(2).upper(),
                        "reason": ""
                    })
                result["theoretical_relations"] = relations
                return result

        return default


# =============================================================================
# CELL 7: RULE-BASED EXTRACTOR (FAST FALLBACK)
# =============================================================================

class RuleBasedExtractor:
    """Fast rule-based classification using keywords with weighted scoring."""

    # Weighted fingerprints: (keyword, weight) - higher weight = more specific
    FINGERPRINTS = {
        'IIT': [
            ('integrated information theory', 3), ('integrated information', 2),
            ('phi', 1), ('tononi', 3), ('cause-effect repertoire', 3),
            ('posterior hot zone', 2), ('maximally irreducible', 3),
            ('exclusion postulate', 3), ('intrinsic causal power', 3),
            ('iit', 2), ('cerebral cortex consciousness', 1)
        ],
        'GNWT': [
            ('global neuronal workspace', 3), ('global workspace theory', 3),
            ('global workspace', 2), ('dehaene', 3), ('baars', 3),
            ('ignition', 2), ('broadcasting', 1), ('p3b', 2), ('p300', 1),
            ('fronto-parietal', 2), ('access consciousness', 2),
            ('prefrontal cortex', 1), ('gnw', 2)
        ],
        'HOT': [
            ('higher-order thought', 3), ('higher-order theory', 3),
            ('higher-order', 2), ('rosenthal', 3), ('lau hakwan', 3),
            ('metacognition', 2), ('meta-cognition', 2),
            ('awareness of awareness', 3), ('second-order', 2),
            ('prefrontal metacognition', 2)
        ],
        'RPT': [
            ('recurrent processing theory', 3), ('recurrent processing', 2),
            ('lamme', 3), ('local recurrence', 3), ('feedforward sweep', 2),
            ('reentrant processing', 2), ('re-entrant', 2), ('fahrenfort', 3),
            ('visual cortex recurrence', 3), ('v1 feedback', 2)
        ]
    }

    # Exclusion patterns that reduce scores
    EXCLUSIONS = {
        'RPT': ['prefrontal cortex', 'global workspace', 'frontal'],
        'IIT': ['global workspace', 'gnw'],
        'GNWT': ['integrated information', 'posterior hot zone']
    }

    def extract(self, abstract: str) -> Dict:
        if pd.isna(abstract):
            return self._default()

        text = ' ' + str(abstract).lower() + ' '

        # Calculate weighted scores
        scores = {}
        for theory, patterns in self.FINGERPRINTS.items():
            score = sum(weight for keyword, weight in patterns if keyword in text)
            scores[theory] = score

        # Apply exclusions
        for theory, exclusions in self.EXCLUSIONS.items():
            if scores.get(theory, 0) > 0:
                for excl in exclusions:
                    if excl in text:
                        scores[theory] = max(0, scores[theory] - 2)

        # Require minimum score threshold
        max_score = max(scores.values()) if scores else 0
        if max_score >= 2:  # Require at least 2 points
            theory = max(scores, key=scores.get)
            confidence = min(max_score / 5, 1.0)
        else:
            theory = "Neutral"
            confidence = 0.0

        # Paradigm
        paradigm = "No-Report" if any(x in text for x in ['no-report', 'implicit', 'no report']) else "Report"

        # Type
        type_val = "State" if any(x in text for x in ['anesthesia', 'sleep', 'wake', 'coma']) else "Content"

        # Tone
        tone = "Neutral"
        if any(x in text for x in ['refutes', 'falsifies', 'disproves']):
            tone = "Dismissive"
        elif any(x in text for x in ['challenges', 'questions', 'problematic']):
            tone = "Critical"
        elif any(x in text for x in ['integrates', 'bridges', 'complementary']):
            tone = "Constructive"

        return {
            "theory": theory, "paradigm": paradigm, "type": type_val,
            "epistemic": "Post-Hoc", "anatomy": [], "tone": tone,
            "target": "Function", "subject": "Human", "confidence": confidence,
            "fingerprints": [], "dogmatism_score": 5, "_extraction_failed": False,
            "debate_used": False
        }

    def _default(self):
        return {
            "theory": "Neutral", "paradigm": "Report", "type": "Content",
            "epistemic": "Post-Hoc", "anatomy": [], "tone": "Neutral",
            "target": "Function", "subject": "Human", "confidence": 0.0,
            "fingerprints": [], "dogmatism_score": 5, "_extraction_failed": True,
            "debate_used": False
        }

    def batch_extract(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"\n{'='*60}\n📝 RULE-BASED EXTRACTION (FAST)\n{'='*60}")
        results = [self.extract(row['Abstract']) for _, row in tqdm(df.iterrows(), total=len(df))]
        df_out = df.copy()
        for k in results[0].keys():
            df_out[k] = [r[k] for r in results]

        print(f"\n📊 Theory Distribution:")
        for t, c in df_out['theory'].value_counts().items():
            print(f"   {t}: {c} ({c/len(df_out)*100:.1f}%)")
        return df_out


# =============================================================================
# CELL 8: GEMINI EMBEDDER
# =============================================================================

class GeminiEmbedder:
    def __init__(self, api_key: str, model: str = EMBEDDING_MODEL):
        self.api_key = api_key
        self.model = model
        self.batch_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
        self.session = requests.Session()

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[Optional[np.ndarray]]:
        print(f"\n{'='*60}\n🧮 GENERATING EMBEDDINGS\n{'='*60}")
        embeddings = [None] * len(texts)

        for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[start:start + batch_size]
            reqs = []
            idxs = []

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
                        time.sleep(30)
                        continue
                    r.raise_for_status()

                    for j, emb in enumerate(r.json().get('embeddings', [])):
                        if 'values' in emb:
                            embeddings[idxs[j]] = np.array(emb['values'])
                    break
                except Exception as e:
                    time.sleep(5)

            time.sleep(0.3)

        count = sum(1 for e in embeddings if e is not None)
        print(f"✅ Embedded: {count}/{len(texts)}")
        return embeddings


# =============================================================================
# CELL 9: BERTOPIC ANALYSIS (v5.0 Feature)
# =============================================================================

def run_bertopic_analysis(df: pd.DataFrame, output_folder: str) -> Dict:
    """
    BERTopic dynamic topic modeling with topics_over_time.
    Finds micro-topics within theories and tracks semantic shifts.
    """
    print(f"\n{'='*60}\n🔬 BERTOPIC TOPIC MODELING (v5.0)\n{'='*60}")

    if not BERTOPIC_AVAILABLE:
        print("⚠️ BERTopic not installed. Run: pip install bertopic")
        return {"error": "BERTopic not available"}

    if not MATPLOTLIB_AVAILABLE:
        print("⚠️ matplotlib not installed")
        return {"error": "matplotlib not available"}

    results = {}

    # Prepare documents
    docs = df['Abstract'].fillna("").tolist()
    timestamps = df['Year'].tolist()
    theories = df['theory'].tolist() if 'theory' in df.columns else None

    print(f"Documents: {len(docs)}")

    try:
        # Create BERTopic model with proper stopword removal
        print("Creating BERTopic model...")

        # Custom vectorizer with stopwords removed and scientific terms preserved
        vectorizer_model = CountVectorizer(
            stop_words='english',
            min_df=5,
            max_df=0.95,
            ngram_range=(1, 2)
        )

        # Use KeyBERTInspired for better topic representations
        representation_model = KeyBERTInspired()

        topic_model = BERTopic(
            language="english",
            min_topic_size=15,
            nr_topics="auto",
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            verbose=True
        )

        # Fit model
        topics, probs = topic_model.fit_transform(docs)

        # Get topic info
        topic_info = topic_model.get_topic_info()
        print(f"\n📊 Found {len(topic_info) - 1} topics (excluding outlier topic)")

        # Save top topics
        results['topics'] = []
        for _, row in topic_info.head(20).iterrows():
            if row['Topic'] != -1:
                topic_words = topic_model.get_topic(row['Topic'])
                results['topics'].append({
                    'topic_id': row['Topic'],
                    'count': row['Count'],
                    'words': [w for w, _ in topic_words[:10]]
                })
                print(f"  Topic {row['Topic']}: {', '.join([w for w, _ in topic_words[:5]])}")

        # Topics over time
        print("\n📈 Analyzing topics over time...")
        topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=10)

        # Save visualization
        fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
        fig.write_html(os.path.join(output_folder, "13_bertopic_over_time.html"))
        print("  ✅ Saved: 13_bertopic_over_time.html")

        # Topics per class (if theories available)
        if theories and len(set(theories)) > 1:
            print("\n🏷️ Topics per theory...")
            topics_per_class = topic_model.topics_per_class(docs, classes=theories)

            fig = topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=10)
            fig.write_html(os.path.join(output_folder, "13b_bertopic_per_theory.html"))
            print("  ✅ Saved: 13b_bertopic_per_theory.html")

            results['topics_per_theory'] = topics_per_class.to_dict()

        # Hierarchical topics
        print("\n🌳 Creating topic hierarchy...")
        hierarchical_topics = topic_model.hierarchical_topics(docs)
        fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        fig.write_html(os.path.join(output_folder, "13c_bertopic_hierarchy.html"))
        print("  ✅ Saved: 13c_bertopic_hierarchy.html")

        results['success'] = True

    except Exception as e:
        print(f"⚠️ BERTopic error: {e}")
        results['error'] = str(e)

    return results


# =============================================================================
# CELL 10: ARGUMENTATION MINING (v5.0 Feature)
# =============================================================================

def run_argument_mining(df: pd.DataFrame, api_key: str, output_folder: str,
                        sample_size: int = 500) -> Dict:
    """
    Extract argumentative structure: Claims, Premises, Support/Attack relations.
    """
    print(f"\n{'='*60}\n⚖️ ARGUMENTATION MINING (v5.0)\n{'='*60}")

    if not api_key:
        print("⚠️ API key required for argument mining")
        return {"error": "No API key"}

    # Sample papers (full extraction is slow)
    if len(df) > sample_size:
        print(f"Sampling {sample_size} papers for argument mining...")
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df

    extractor = GeminiExtractor(api_key, GEMINI_MODEL)

    results = {
        'arguments': [],
        'support_graph': defaultdict(list),
        'attack_graph': defaultdict(list),
        'theory_support_counts': defaultdict(int),
        'theory_attack_counts': defaultdict(int)
    }

    print(f"Extracting arguments from {len(sample_df)} papers...")

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Mining"):
        try:
            arg_data = extractor.extract_arguments(row['Abstract'])

            if arg_data['main_claim']:
                results['arguments'].append({
                    'paper_idx': idx,
                    'title': row.get('Title', '')[:100],
                    'year': row.get('Year', 0),
                    'main_claim': arg_data['main_claim'],
                    'premises': arg_data['premises'],
                    'relations': arg_data['theoretical_relations'],
                    'strength': arg_data['argument_strength']
                })

                # Build graph
                for rel in arg_data.get('theoretical_relations', []):
                    if isinstance(rel, dict):
                        relation = rel.get('relation', '').upper()
                        theory = rel.get('theory', '').upper()

                        if 'SUPPORT' in relation:
                            results['support_graph'][theory].append(idx)
                            results['theory_support_counts'][theory] += 1
                        elif 'ATTACK' in relation:
                            results['attack_graph'][theory].append(idx)
                            results['theory_attack_counts'][theory] += 1

            time.sleep(API_DELAY)

        except Exception as e:
            if DEBUG_MODE:
                print(f"⚠️ Argument mining error at {idx}: {e}")

    # Summary
    print(f"\n📊 Argumentation Summary:")
    print(f"   Papers with arguments extracted: {len(results['arguments'])}")

    print(f"\n   Support counts by theory:")
    for theory, count in sorted(results['theory_support_counts'].items(), key=lambda x: -x[1]):
        print(f"     {theory}: {count}")

    print(f"\n   Attack counts by theory:")
    for theory, count in sorted(results['theory_attack_counts'].items(), key=lambda x: -x[1]):
        print(f"     {theory}: {count}")

    # Visualize argument graph
    if MATPLOTLIB_AVAILABLE and results['arguments']:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Support bar chart
            theories = ['IIT', 'GNWT', 'HOT', 'RPT']
            supports = [results['theory_support_counts'].get(t, 0) for t in theories]
            attacks = [results['theory_attack_counts'].get(t, 0) for t in theories]

            x = np.arange(len(theories))
            width = 0.35

            axes[0].bar(x - width/2, supports, width, label='Supports', color='#2A9D8F')
            axes[0].bar(x + width/2, attacks, width, label='Attacks', color='#E63946')
            axes[0].set_ylabel('Number of Papers')
            axes[0].set_title('Argumentative Stance by Theory')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(theories)
            axes[0].legend()

            # Support/Attack ratio
            ratios = []
            for t in theories:
                s = results['theory_support_counts'].get(t, 0)
                a = results['theory_attack_counts'].get(t, 0)
                ratio = s / (s + a) if (s + a) > 0 else 0.5
                ratios.append(ratio)

            colors = ['#2A9D8F' if r > 0.5 else '#E63946' for r in ratios]
            axes[1].bar(theories, ratios, color=colors)
            axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
            axes[1].set_ylabel('Support Ratio (>0.5 = more support)')
            axes[1].set_title('Support vs Attack Ratio by Theory')
            axes[1].set_ylim(0, 1)

            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, "14_argument_mining.png"), dpi=300)
            plt.close()
            print("  ✅ Saved: 14_argument_mining.png")

        except Exception as e:
            print(f"⚠️ Visualization error: {e}")

    # Save arguments to JSON
    with open(os.path.join(output_folder, "14_arguments.json"), 'w') as f:
        json.dump({
            'arguments': results['arguments'][:100],  # Save first 100
            'support_counts': dict(results['theory_support_counts']),
            'attack_counts': dict(results['theory_attack_counts'])
        }, f, indent=2, default=str)
    print("  ✅ Saved: 14_arguments.json")

    return results


# =============================================================================
# CELL 11: SEMANTIC CLUSTERING
# =============================================================================

def run_semantic_clustering(df: pd.DataFrame, embeddings: List, output_folder: str) -> Dict:
    print(f"\n{'='*60}\n🌌 SEMANTIC CLUSTERING\n{'='*60}")

    if not UMAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        print("⚠️ Missing umap-learn or matplotlib")
        return {"error": "Missing dependencies"}

    valid_mask = [e is not None for e in embeddings]
    valid_embeddings = [e for e in embeddings if e is not None]

    if len(valid_embeddings) < 50:
        return {"error": "Insufficient embeddings"}

    X = np.vstack(valid_embeddings)
    valid_df = df[valid_mask].copy().reset_index(drop=True)

    print(f"Clustering {len(X)} embeddings...")

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    X_2d = reducer.fit_transform(X)

    valid_df['umap_x'], valid_df['umap_y'] = X_2d[:, 0], X_2d[:, 1]

    # Plot
    plt.figure(figsize=(14, 10))
    colors = {'IIT': '#E63946', 'GNWT': '#457B9D', 'HOT': '#2A9D8F', 'RPT': '#E9C46A', 'Neutral': '#CCCCCC'}

    for theory in ['Neutral', 'IIT', 'GNWT', 'HOT', 'RPT']:
        mask = valid_df['theory'] == theory
        if mask.sum() > 0:
            alpha = 0.3 if theory == 'Neutral' else 0.8
            size = 20 if theory == 'Neutral' else 50
            plt.scatter(valid_df.loc[mask, 'umap_x'], valid_df.loc[mask, 'umap_y'],
                       c=colors[theory], label=f"{theory} (n={mask.sum()})",
                       alpha=alpha, s=size)

    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('Semantic Space by Theory')
    plt.legend()
    plt.savefig(os.path.join(output_folder, "8_semantic_space.png"), dpi=300)
    plt.close()
    print("  ✅ Saved: 8_semantic_space.png")

    # Centroids
    centroids = {}
    for theory in ['IIT', 'GNWT', 'HOT', 'RPT', 'Neutral']:
        mask = valid_df['theory'] == theory
        if mask.sum() >= 3:
            centroids[theory] = np.mean(X[mask], axis=0)

    return {"valid_df": valid_df, "centroids": centroids, "X": X}


# =============================================================================
# CELL 12: ORIGINAL ANALYSES (1-7, 9-12) - Condensed
# =============================================================================

def run_contrast_audit(df, out):
    print(f"\n{'='*60}\n📊 ANALYSIS 1: CONTRAST\n{'='*60}")
    dft = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])]
    if len(dft) < 20:
        return {"error": "Insufficient data"}

    counts = dft.groupby(['paradigm', 'theory']).size().reset_index(name='count')
    paradigms = dft['paradigm'].unique().tolist()
    theories = dft['theory'].unique().tolist()

    fig = go.Figure(go.Sankey(
        node=dict(label=paradigms + theories),
        link=dict(
            source=[paradigms.index(r['paradigm']) for _, r in counts.iterrows()],
            target=[len(paradigms) + theories.index(r['theory']) for _, r in counts.iterrows()],
            value=counts['count'].tolist()
        )
    ))
    fig.update_layout(title="ConTraSt: Paradigm → Theory")
    fig.write_image(os.path.join(out, "1_contrast.png"), scale=2)
    print("  ✅ Saved: 1_contrast.png")
    return {"n_papers": len(dft)}


def run_moving_goalposts(df, out):
    print(f"\n{'='*60}\n📈 ANALYSIS 2: MOVING GOALPOSTS\n{'='*60}")
    if not SBERT_AVAILABLE:
        return {"error": "sentence-transformers not available"}

    model = SentenceTransformer('all-MiniLM-L6-v2')
    dft = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])].copy()

    colors = {'IIT': '#E63946', 'GNWT': '#457B9D', 'HOT': '#2A9D8F', 'RPT': '#E9C46A'}
    fig = go.Figure()

    for theory in colors:
        tdf = dft[dft['theory'] == theory]
        if len(tdf) < 5:
            continue

        yearly = {}
        for year in range(MIN_YEAR, MAX_YEAR + 1):
            texts = tdf[tdf['Year'] == year]['Abstract'].tolist()
            if len(texts) >= 2:
                yearly[year] = model.encode([t[:500] for t in texts])

        if len(yearly) < 3:
            continue

        years = sorted(yearly.keys())
        base = BASELINE_YEAR if BASELINE_YEAR in years else years[0]
        base_emb = np.mean(yearly[base], axis=0)

        drift = [cosine(base_emb, np.mean(yearly[y], axis=0)) for y in years]
        fig.add_trace(go.Scatter(x=years, y=drift, name=theory, line=dict(color=colors[theory])))

    fig.update_layout(title="Semantic Drift", xaxis_title="Year", yaxis_title="Drift")
    fig.write_image(os.path.join(out, "2_goalposts.png"), scale=2)
    print("  ✅ Saved: 2_goalposts.png")
    return {}


def run_adversarial(df, out):
    print(f"\n{'='*60}\n⚔️ ANALYSIS 3: ADVERSARIAL INDEX\n{'='*60}")
    patterns = {
        'IIT': r'\b(integrated information|iit\b|tononi)',
        'GNWT': r'\b(global.?workspace|gnw\b|dehaene)',
        'HOT': r'\b(higher.?order|rosenthal)',
        'RPT': r'\b(recurrent processing|lamme)'
    }

    df['n_theories'] = df['Abstract'].apply(
        lambda x: sum(1 for p in patterns.values() if re.search(p, str(x).lower())) if pd.notna(x) else 0
    )
    df['adversarial'] = df['n_theories'] >= 2

    yearly = df.groupby('Year').agg(total=('Title', 'count'), adv=('adversarial', 'sum')).reset_index()
    yearly['pct'] = yearly['adv'] / yearly['total'] * 100
    yearly = yearly[(yearly['Year'] >= MIN_YEAR) & (yearly['Year'] <= MAX_YEAR)]

    z = np.polyfit(yearly['Year'], yearly['pct'], 1)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly['Year'], y=yearly['pct'], marker_color='#E63946'))
    fig.add_trace(go.Scatter(x=yearly['Year'], y=np.poly1d(z)(yearly['Year']), mode='lines'))
    fig.update_layout(title=f"Adversarial Index (Trend: {'↑' if z[0]>0 else '↓'})")
    fig.write_image(os.path.join(out, "3_adversarial.png"), scale=2)
    print(f"  Total: {df['adversarial'].sum()} | ✅ Saved: 3_adversarial.png")
    return {"total": int(df['adversarial'].sum())}


def run_aggression(df, out):
    print(f"\n{'='*60}\n😤 ANALYSIS 5: AGGRESSION INDEX\n{'='*60}")
    dft = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])].copy()
    if len(dft) < 10:
        return {"error": "Insufficient data"}

    dft['aggressive'] = dft['tone'].isin(['Dismissive', 'Critical'])
    yearly = dft.groupby('Year').agg(total=('Title', 'count'), agg=('aggressive', 'sum')).reset_index()
    yearly['pct'] = yearly['agg'] / yearly['total'] * 100
    yearly = yearly[(yearly['Year'] >= MIN_YEAR) & (yearly['Year'] <= MAX_YEAR)]

    if len(yearly) < 3:
        return {"error": "Insufficient data"}

    z = np.polyfit(yearly['Year'], yearly['pct'], 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly['Year'], y=yearly['pct'], mode='lines+markers', line=dict(color='#E63946')))
    fig.add_trace(go.Scatter(x=yearly['Year'], y=np.poly1d(z)(yearly['Year']), mode='lines', line=dict(dash='dash')))
    fig.update_layout(title=f"Aggression Index (Trend: {'↑' if z[0]>0 else '↓'})")
    fig.write_image(os.path.join(out, "5_aggression.png"), scale=2)
    print(f"  ✅ Saved: 5_aggression.png")
    return {"trend": float(z[0])}


def run_methodology_predicts_theory(df, out):
    print(f"\n{'='*60}\n🔬 METHODOLOGY PREDICTS THEORY\n{'='*60}")
    dft = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])].copy()
    if len(dft) < 50:
        return {"error": "Insufficient data"}

    le = {col: LabelEncoder() for col in ['paradigm', 'type', 'subject', 'target', 'theory']}
    for col in le:
        dft[f'{col}_enc'] = le[col].fit_transform(dft[col])

    X = dft[['paradigm_enc', 'type_enc', 'subject_enc', 'target_enc']].values
    y = dft['theory_enc'].values

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = cross_val_score(rf, X, y, cv=5)
    print(f"  CV Accuracy: {cv.mean():.3f} ± {cv.std():.3f}")

    rf.fit(X, y)
    imp = dict(zip(['Paradigm', 'Type', 'Subject', 'Target'], rf.feature_importances_))
    print(f"  Importances: {imp}")

    return {"cv_accuracy": float(cv.mean()), "importances": imp}


# =============================================================================
# CELL 13: MAIN PIPELINE
# =============================================================================

def run_full_audit(csv_path: str = CSV_PATH, api_key: str = GEMINI_API_KEY,
                   output_folder: str = OUTPUT_FOLDER, use_llm: bool = True,
                   use_debate: bool = False, run_embeddings: bool = True) -> Dict:
    """Run the full v5.0 audit pipeline."""

    print(f"\n{'='*70}")
    print("    🧠 CONSCIOUSNESS WARS AUDIT v5.0 🧠")
    print(f"{'='*70}")

    os.makedirs(output_folder, exist_ok=True)
    df = load_bibliography(csv_path)

    # Classification
    if use_llm and api_key:
        try:
            extractor = GeminiExtractor(api_key, GEMINI_MODEL)
            df_classified = extractor.batch_extract_parallel(df, MAX_WORKERS, use_debate=use_debate)
            if df_classified is None:
                print("⚠️ LLM extraction failed, using rule-based...")
                df_classified = RuleBasedExtractor().batch_extract(df)
        except Exception as e:
            print(f"⚠️ LLM error: {e}")
            df_classified = RuleBasedExtractor().batch_extract(df)
    else:
        df_classified = RuleBasedExtractor().batch_extract(df)

    results = {}

    # Original analyses
    for name, fn in [('1_contrast', run_contrast_audit), ('2_goalposts', run_moving_goalposts),
                     ('3_adversarial', run_adversarial), ('5_aggression', run_aggression),
                     ('10_methodology', run_methodology_predicts_theory)]:
        try:
            results[name] = fn(df_classified, output_folder)
        except Exception as e:
            print(f"⚠️ {name} failed: {e}")

    # Embeddings and clustering
    if run_embeddings and api_key:
        try:
            embeddings = GeminiEmbedder(api_key).embed_batch(df_classified['Abstract'].tolist())
            cluster_result = run_semantic_clustering(df_classified, embeddings, output_folder)
            if 'error' not in cluster_result:
                results['8_clustering'] = cluster_result
        except Exception as e:
            print(f"⚠️ Clustering failed: {e}")

    # v5.0 Features
    if ENABLE_BERTOPIC and BERTOPIC_AVAILABLE:
        try:
            results['13_bertopic'] = run_bertopic_analysis(df_classified, output_folder)
        except Exception as e:
            print(f"⚠️ BERTopic failed: {e}")

    if ENABLE_ARGUMENT_MINING and api_key:
        try:
            results['14_arguments'] = run_argument_mining(df_classified, api_key, output_folder)
        except Exception as e:
            print(f"⚠️ Argument mining failed: {e}")

    # Save
    try:
        df_classified.to_csv(os.path.join(output_folder, "classified_papers.csv"), index=False)
    except PermissionError:
        df_classified.to_csv(os.path.join(output_folder, f"classified_{int(time.time())}.csv"), index=False)

    with open(os.path.join(output_folder, "results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*70}")
    print("    ✅ AUDIT COMPLETE")
    print(f"{'='*70}")
    print(f"Papers: {len(df_classified)}")
    for t, c in df_classified['theory'].value_counts().items():
        print(f"   {t}: {c} ({c/len(df_classified)*100:.1f}%)")
    print(f"\n📁 Results: {output_folder}")

    return {'data': df_classified, 'results': results}


def test_api_connection(api_key: str, model: str) -> bool:
    """Test API connection."""
    print(f"\n{'='*60}\n🔌 TESTING API\n{'='*60}")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    try:
        r = requests.post(f"{url}?key={api_key}", json={
            "contents": [{"parts": [{"text": "Say OK"}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 10}
        }, timeout=30)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print("✅ API OK")
            return True
        print(f"❌ Error: {r.text[:200]}")
    except Exception as e:
        print(f"❌ {e}")
    return False


if __name__ == "__main__":
    print("=" * 70)
    print("    CONSCIOUSNESS WARS AUDIT v5.0")
    print("=" * 70)

    if not GEMINI_API_KEY:
        print("\n❌ GEMINI_API_KEY not found in .env")
        exit(1)

    print(f"\n🤖 Model: {GEMINI_MODEL}")
    print(f"📁 CSV: {CSV_PATH}")

    if not test_api_connection(GEMINI_API_KEY, GEMINI_MODEL):
        print("\n❌ API failed. Check model name.")
        print("Valid: gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro")
        exit(1)

    # Run with debate mode disabled by default (faster)
    results = run_full_audit(use_debate=False)
    print("\n✅ Done!")
