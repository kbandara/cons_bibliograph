"""
================================================================================
CONSCIOUSNESS WARS BIBLIOMETRIC AUDIT PIPELINE
================================================================================
A comprehensive bibliometric analysis of consciousness theories:
IIT (Integrated Information Theory), GNWT (Global Neuronal Workspace Theory),
HOT (Higher-Order Theory), and RPT (Recurrent Processing Theory).

Three Analyses:
1. ConTraSt Audit - Replication of Yaron et al. (2022)
2. Moving Goalposts - Concept drift analysis over time
3. Adversarial Index - Cross-theory citation patterns

For use in Google Colab with Gemini API
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

# Your Gemini API Key (get one at https://makersuite.google.com/app/apikey)
GEMINI_API_KEY = ""  # <-- PASTE YOUR API KEY HERE

# Google Drive folder path for saving outputs (will be created if doesn't exist)
GDRIVE_OUTPUT_FOLDER = "/content/drive/MyDrive/consciousness_audit_results"

# Path to your bibliography CSV file
CSV_PATH = "/content/cons_bib.csv"  # Update if your file is elsewhere

# Gemini model to use
GEMINI_MODEL = "gemini-1.5-flash"  # Options: "gemini-1.5-flash", "gemini-1.5-pro"

# Parallel processing settings
MAX_WORKERS = 10  # Number of parallel API requests (be careful with rate limits)
API_DELAY = 0.1   # Delay between batches (seconds)

# Analysis settings
BASELINE_YEAR = 2005  # Reference year for semantic drift calculation
MIN_YEAR = 2000       # Earliest year to include in analysis
MAX_YEAR = 2025       # Latest year to include in analysis

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
        'kaleido',
        'google-generativeai',
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
# CELL 3: MOUNT GOOGLE DRIVE
# =============================================================================

def mount_google_drive():
    """Mount Google Drive for saving outputs."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully!")
        return True
    except ImportError:
        print("⚠️ Not running in Google Colab - skipping Drive mount")
        return False
    except Exception as e:
        print(f"⚠️ Failed to mount Google Drive: {e}")
        return False

def setup_output_folder(folder_path: str) -> str:
    """Create output folder if it doesn't exist."""
    import os
    os.makedirs(folder_path, exist_ok=True)
    print(f"📁 Output folder: {folder_path}")
    return folder_path

# Uncomment to mount Drive:
# mount_google_drive()
# setup_output_folder(GDRIVE_OUTPUT_FOLDER)

# =============================================================================
# CELL 4: IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import json
import re
import time
import os
import warnings
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

import requests
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# CELL 5: DATA LOADING
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

        return df_valid

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
# CELL 6: GEMINI LLM EXTRACTOR (PARALLEL)
# =============================================================================

class GeminiExtractor:
    """
    Extract structured information from abstracts using Google Gemini API.
    Supports parallel processing for faster extraction.
    """

    EXTRACTION_PROMPT = '''You are a consciousness science expert. Analyze this research abstract and classify it.

ABSTRACT:
{abstract}

Based on the abstract, determine:

1. THEORY - Which consciousness theory does this paper primarily support or investigate?
   - IIT: Integrated Information Theory (Tononi). Look for: phi (Φ), integrated information, information geometry, exclusion postulate, intrinsic causal power, IIT, qualia space
   - GNWT: Global Neuronal Workspace Theory (Dehaene/Baars). Look for: global workspace, ignition, broadcasting, access consciousness, frontal-parietal, P3b, late positivity
   - HOT: Higher-Order Theory (Rosenthal/Lau). Look for: higher-order thought, higher-order representation, metacognition, prefrontal cortex for awareness
   - RPT: Recurrent Processing Theory (Lamme). Look for: recurrent processing, feedback loops, local recurrence, V1, feedforward vs feedback
   - Neutral: If the paper doesn't clearly align with any specific theory, or is methodological/review without theoretical commitment

2. PARADIGM - What experimental approach is used?
   - Report: Participants explicitly report their conscious experience (verbal reports, button presses indicating awareness)
   - No-Report: Consciousness is inferred without requiring explicit reports (e.g., physiological measures, no-report paradigms)

3. TYPE - What aspect of consciousness is studied?
   - Content: Studies WHAT we are conscious of (specific perceptions, visual features, objects)
   - State: Studies WHETHER/HOW MUCH consciousness is present (levels, anesthesia, sleep, disorders of consciousness)

4. EPISTEMIC - What is the hypothesis approach?
   - A Priori: Predictions derived from theory before seeing the data
   - Post-Hoc: Exploratory analyses, interpretations made after seeing results

Respond with ONLY this exact JSON format, nothing else:
{{"theory": "IIT", "paradigm": "Report", "type": "Content", "epistemic": "A Priori", "confidence": 0.8}}

Replace the values with your classifications. Confidence should be 0.0-1.0 based on how clear the classification is.'''

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.session = requests.Session()

        # Validate API key
        if not api_key or api_key == "":
            raise ValueError("❌ GEMINI_API_KEY is not set! Please add your API key in the Configuration section.")

    def _call_gemini(self, abstract: str) -> Dict[str, Any]:
        """Make a single Gemini API call."""
        prompt = self.EXTRACTION_PROMPT.format(abstract=abstract[:3000])

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 150,
                "topP": 0.8,
            }
        }

        headers = {
            "Content-Type": "application/json",
        }

        try:
            response = self.session.post(
                f"{self.base_url}?key={self.api_key}",
                json=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 429:
                # Rate limited - wait and signal retry
                time.sleep(2)
                return {"_retry": True}

            response.raise_for_status()
            result = response.json()

            # Extract text from response
            if 'candidates' in result and result['candidates']:
                text = result['candidates'][0]['content']['parts'][0]['text']
                return self._parse_response(text)

            return self._default()

        except requests.exceptions.RequestException as e:
            return self._default()

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from Gemini response with robust extraction."""
        try:
            # Clean the text - remove markdown code blocks if present
            text = text.strip()
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'^```\s*', '', text)
            text = re.sub(r'\s*```$', '', text)

            # Find JSON object in response
            json_match = re.search(r'\{[^{}]*"theory"[^{}]*\}', text, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)

                # Validate and normalize values
                theory = str(data.get("theory", "Neutral")).upper()
                if theory not in ["IIT", "GNWT", "HOT", "RPT", "NEUTRAL"]:
                    theory = "Neutral"
                else:
                    theory = theory.title() if theory == "NEUTRAL" else theory

                paradigm = str(data.get("paradigm", "Report"))
                if "no" in paradigm.lower():
                    paradigm = "No-Report"
                else:
                    paradigm = "Report"

                content_type = str(data.get("type", "Content"))
                if "state" in content_type.lower():
                    content_type = "State"
                else:
                    content_type = "Content"

                epistemic = str(data.get("epistemic", "Post-Hoc"))
                if "priori" in epistemic.lower():
                    epistemic = "A Priori"
                else:
                    epistemic = "Post-Hoc"

                confidence = float(data.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))

                return {
                    "theory": theory,
                    "paradigm": paradigm,
                    "type": content_type,
                    "epistemic": epistemic,
                    "confidence": confidence
                }

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            pass

        # If JSON parsing failed, try to extract from text directly
        return self._extract_from_text(text)

    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """Fallback: extract classifications from raw text."""
        text_upper = text.upper()

        # Theory detection
        theory = "Neutral"
        for t in ["IIT", "GNWT", "HOT", "RPT"]:
            if t in text_upper and "NEUTRAL" not in text_upper:
                theory = t
                break

        # Paradigm detection
        paradigm = "No-Report" if "NO-REPORT" in text_upper or "NO REPORT" in text_upper else "Report"

        # Type detection
        content_type = "State" if "STATE" in text_upper else "Content"

        # Epistemic detection
        epistemic = "A Priori" if "A PRIORI" in text_upper or "APRIORI" in text_upper else "Post-Hoc"

        return {
            "theory": theory,
            "paradigm": paradigm,
            "type": content_type,
            "epistemic": epistemic,
            "confidence": 0.3
        }

    def _default(self) -> Dict[str, Any]:
        """Return default extraction when API fails."""
        return {
            "theory": "Neutral",
            "paradigm": "Report",
            "type": "Content",
            "epistemic": "Post-Hoc",
            "confidence": 0.0
        }

    def extract_single(self, abstract: str, max_retries: int = 3) -> Dict[str, Any]:
        """Extract from a single abstract with retries."""
        if pd.isna(abstract) or len(str(abstract)) < 50:
            return self._default()

        for attempt in range(max_retries):
            result = self._call_gemini(abstract)
            if result.get("_retry"):
                time.sleep(1 * (attempt + 1))
                continue
            return result

        return self._default()

    def batch_extract_parallel(
        self,
        df: pd.DataFrame,
        abstract_col: str = 'Abstract',
        max_workers: int = 10,
        delay: float = 0.1
    ) -> pd.DataFrame:
        """Extract from all abstracts using parallel processing."""
        print("\n" + "=" * 60)
        print("🤖 EXTRACTING PAPER CLASSIFICATIONS WITH GEMINI")
        print("=" * 60)
        print(f"Model: {self.model}")
        print(f"Papers to process: {len(df)}")
        print(f"Parallel workers: {max_workers}")

        abstracts = df[abstract_col].tolist()
        indices = df.index.tolist()
        results = [None] * len(abstracts)

        # Process in parallel with progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.extract_single, abstract): i
                for i, abstract in enumerate(abstracts)
            }

            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_idx), total=len(abstracts), desc="Extracting"):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = self._default()

                # Small delay to avoid rate limiting
                time.sleep(delay)

        # Add results to DataFrame
        result_df = df.copy()
        for key in ['theory', 'paradigm', 'type', 'epistemic', 'confidence']:
            result_df[key] = [r[key] for r in results]

        # Print statistics
        print(f"\n✅ Extraction complete!")
        print(f"\n📊 Theory distribution:")
        theory_counts = result_df['theory'].value_counts()
        for theory, count in theory_counts.items():
            pct = count / len(result_df) * 100
            print(f"   {theory}: {count} ({pct:.1f}%)")

        return result_df

# =============================================================================
# CELL 7: RULE-BASED EXTRACTOR (FALLBACK)
# =============================================================================

class RuleBasedExtractor:
    """Rule-based extraction fallback using keyword matching."""

    THEORY_KEYWORDS = {
        'IIT': [
            'integrated information theory', 'integrated information', 'phi', 'tononi',
            'information integration', 'exclusion', 'intrinsic', 'causal power', ' iit ',
            'qualia space', 'compositional', 'intrinsic cause-effect', 'axioms of consciousness'
        ],
        'GNWT': [
            'global workspace', 'global neuronal workspace', 'dehaene', 'baars',
            'ignition', 'broadcasting', 'access consciousness', 'gnwt', ' gnw ',
            'frontal-parietal', 'frontoparietal', 'prefrontal broadcasting', 'late positivity'
        ],
        'HOT': [
            'higher-order thought', 'higher-order theory', 'higher order', 'rosenthal',
            'metacognition', 'hot theory', 'higher-order representation', 'lau',
            'meta-awareness', 'first-order representation'
        ],
        'RPT': [
            'recurrent processing theory', 'recurrent processing', 'lamme', 'feedback loops',
            'local recurrence', 'feedforward', 'recurrent feedback', ' rpt ',
            'feedforward sweep', 'recurrent activity'
        ]
    }

    def extract(self, abstract: str) -> Dict[str, Any]:
        """Extract classifications using keyword matching."""
        if pd.isna(abstract) or not abstract:
            return self._default()

        abstract_lower = ' ' + abstract.lower() + ' '

        # Score each theory
        scores = {}
        for theory, keywords in self.THEORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in abstract_lower)
            scores[theory] = score

        max_score = max(scores.values())
        theory = max(scores, key=scores.get) if max_score > 0 else "Neutral"

        # Simple paradigm detection
        paradigm = "No-Report" if any(kw in abstract_lower for kw in ['no-report', 'no report', 'without report', 'unreportable']) else "Report"

        # Simple type detection
        type_kw = ['level of consciousness', 'disorders of consciousness', 'anesthesia', 'coma', 'vegetative', 'sleep', 'arousal']
        content_type = "State" if any(kw in abstract_lower for kw in type_kw) else "Content"

        # Epistemic detection
        epistemic = "A Priori" if any(kw in abstract_lower for kw in ['predict', 'hypothesis', 'theory-driven', 'a priori']) else "Post-Hoc"

        return {
            "theory": theory,
            "paradigm": paradigm,
            "type": content_type,
            "epistemic": epistemic,
            "confidence": min(max_score / 2.0, 1.0) if max_score > 0 else 0.2
        }

    def _default(self) -> Dict[str, Any]:
        return {"theory": "Neutral", "paradigm": "Report", "type": "Content", "epistemic": "Post-Hoc", "confidence": 0.0}

    def batch_extract(self, df: pd.DataFrame, abstract_col: str = 'Abstract') -> pd.DataFrame:
        """Extract from all abstracts using rules."""
        print("\n" + "=" * 60)
        print("📝 EXTRACTING WITH RULE-BASED CLASSIFIER")
        print("=" * 60)

        results = [self.extract(row[abstract_col]) for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting")]

        result_df = df.copy()
        for key in ['theory', 'paradigm', 'type', 'epistemic', 'confidence']:
            result_df[key] = [r[key] for r in results]

        print(f"\n✅ Extraction complete!")
        print(f"\n📊 Theory distribution:")
        for theory, count in result_df['theory'].value_counts().items():
            print(f"   {theory}: {count} ({count/len(result_df)*100:.1f}%)")

        return result_df

# =============================================================================
# CELL 8: PART 1 - CONTRAST AUDIT
# =============================================================================

def run_contrast_audit(df: pd.DataFrame) -> Tuple[Optional[go.Figure], Dict[str, Any]]:
    """
    PART 1: ConTraSt Audit - Does Methodology Predict Theory?
    """
    print("\n" + "=" * 60)
    print("📊 PART 1: CONTRAST AUDIT")
    print("Does Methodology Predict Theory?")
    print("=" * 60)

    # Filter to papers with theory assignments
    df_theories = df[df['theory'] != 'Neutral'].copy()
    print(f"\nPapers with theory assignment: {len(df_theories)}")

    if len(df_theories) < 10:
        print("⚠️ Insufficient papers with theory classifications for analysis")
        return None, {"error": "Insufficient data", "n_papers": len(df_theories)}

    # Encode features
    le_paradigm = LabelEncoder()
    le_type = LabelEncoder()
    le_theory = LabelEncoder()

    df_theories['paradigm_enc'] = le_paradigm.fit_transform(df_theories['paradigm'])
    df_theories['type_enc'] = le_type.fit_transform(df_theories['type'])
    df_theories['theory_enc'] = le_theory.fit_transform(df_theories['theory'])

    X = df_theories[['paradigm_enc', 'type_enc']].values
    y = df_theories['theory_enc'].values

    # Train RandomForest
    print("\n🌲 Training RandomForest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    cv_scores = cross_val_score(rf, X, y, cv=min(5, len(df_theories)))
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    rf.fit(X, y)
    importances = rf.feature_importances_
    print(f"\nFeature Importances:")
    print(f"   Paradigm: {importances[0]:.3f}")
    print(f"   Type: {importances[1]:.3f}")

    # Create Sankey Diagram
    print("\n🎨 Creating Sankey Diagram...")

    paradigm_theory_counts = df_theories.groupby(['paradigm', 'theory']).size().reset_index(name='count')
    paradigms = df_theories['paradigm'].unique().tolist()
    theories = df_theories['theory'].unique().tolist()

    node_labels = paradigms + theories
    node_colors = ['#2E86AB', '#A23B72'] + ['#F18F01', '#C73E1D', '#3B1F2B', '#95190C'][:len(theories)]

    sources, targets, values, link_colors = [], [], [], []

    for _, row in paradigm_theory_counts.iterrows():
        source_idx = paradigms.index(row['paradigm'])
        target_idx = len(paradigms) + theories.index(row['theory'])
        sources.append(source_idx)
        targets.append(target_idx)
        values.append(row['count'])
        link_colors.append('rgba(46, 134, 171, 0.4)' if row['paradigm'] == 'Report' else 'rgba(162, 59, 114, 0.4)')

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5),
                  label=node_labels, color=node_colors[:len(node_labels)]),
        link=dict(source=sources, target=targets, value=values, color=link_colors)
    )])

    fig.update_layout(
        title=dict(
            text=f"<b>ConTraSt Audit: Paradigm → Theory Flow</b><br><sup>RF Accuracy: {cv_scores.mean():.1%} | n={len(df_theories)} papers</sup>",
            font=dict(size=18)
        ),
        font=dict(size=12), height=500, width=800
    )

    return fig, {
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'feature_importances': {'Paradigm': importances[0], 'Type': importances[1]},
        'n_papers': len(df_theories)
    }

# =============================================================================
# CELL 9: PART 2 - MOVING GOALPOSTS
# =============================================================================

def run_moving_goalposts(df: pd.DataFrame, baseline_year: int = 2005, min_year: int = 2000, max_year: int = 2025) -> Tuple[go.Figure, Dict[str, Any]]:
    """
    PART 2: Moving Goalposts - Concept Drift Analysis
    """
    print("\n" + "=" * 60)
    print("📈 PART 2: MOVING GOALPOSTS ANALYSIS")
    print("Quantifying Semantic Drift Over Time")
    print("=" * 60)

    print("\n🤖 Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    df_theories = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])].copy()
    print(f"Papers with theory assignment: {len(df_theories)}")

    df_theories['definition_text'] = df_theories['Abstract'].apply(lambda x: extract_first_sentences(x, 2))
    df_theories = df_theories[df_theories['definition_text'].str.len() > 20]

    print("\n📊 Computing embeddings by theory and year...")

    drift_data = defaultdict(list)
    colors = {'IIT': '#E63946', 'GNWT': '#457B9D', 'HOT': '#2A9D8F', 'RPT': '#E9C46A'}

    for theory in tqdm(['IIT', 'GNWT', 'HOT', 'RPT'], desc="Processing theories"):
        theory_df = df_theories[df_theories['theory'] == theory]

        if len(theory_df) < 3:
            print(f"  ⚠️ Skipping {theory}: insufficient papers ({len(theory_df)})")
            continue

        embeddings_by_year = {}
        for year in range(min_year, max_year + 1):
            year_df = theory_df[theory_df['Year'] == year]
            if len(year_df) >= 1:
                embeddings_by_year[year] = model.encode(year_df['definition_text'].tolist())

        if not embeddings_by_year:
            continue

        available_years = sorted(embeddings_by_year.keys())
        base_year = baseline_year if baseline_year in available_years else available_years[0]
        baseline_centroid = np.mean(embeddings_by_year[base_year], axis=0)

        for year in available_years:
            year_centroid = np.mean(embeddings_by_year[year], axis=0)
            drift_data[theory].append({
                'year': year,
                'drift': cosine(baseline_centroid, year_centroid),
                'n_papers': len(embeddings_by_year[year])
            })

    print("\n🎨 Creating drift visualization...")

    fig = go.Figure()
    for theory in ['IIT', 'GNWT', 'HOT', 'RPT']:
        if theory not in drift_data:
            continue
        td = pd.DataFrame(drift_data[theory]).sort_values('year')
        if len(td) > 3:
            td['drift_smooth'] = td['drift'].rolling(3, min_periods=1, center=True).mean()
        else:
            td['drift_smooth'] = td['drift']

        fig.add_trace(go.Scatter(
            x=td['year'], y=td['drift_smooth'], mode='lines+markers', name=theory,
            line=dict(color=colors[theory], width=3), marker=dict(size=8)
        ))

    fig.update_layout(
        title=dict(text="<b>Moving Goalposts: Semantic Drift in Theory Definitions</b><br><sup>Cosine distance from baseline (higher = more change)</sup>", font=dict(size=18)),
        xaxis=dict(title="Year", tickmode='linear', tick0=min_year, dtick=5),
        yaxis=dict(title="Semantic Drift (Cosine Distance)", rangemode='tozero'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500, width=900
    )

    fig.add_annotation(text="↑ Rising = Changing definition", xref="paper", yref="paper",
                       x=0.02, y=0.98, showarrow=False, font=dict(size=11, color="gray"))

    return fig, {'drift_data': dict(drift_data), 'baseline_year': baseline_year}

# =============================================================================
# CELL 10: PART 3 - ADVERSARIAL INDEX
# =============================================================================

def run_adversarial_index(df: pd.DataFrame, min_year: int = 2000, max_year: int = 2025) -> Tuple[go.Figure, Dict[str, Any]]:
    """
    PART 3: Adversarial Index - Cross-Theory Engagement
    """
    print("\n" + "=" * 60)
    print("⚔️ PART 3: ADVERSARIAL INDEX")
    print("Measuring Cross-Theory Engagement")
    print("=" * 60)

    theory_patterns = {
        'IIT': r'\b(integrated information( theory)?|iit|tononi|phi\b)',
        'GNWT': r'\b(global (neuronal )?workspace( theory)?|gnw(t)?|dehaene|baars)',
        'HOT': r'\b(higher[- ]order( thought| theory)?|hot theory|rosenthal)',
        'RPT': r'\b(recurrent processing( theory)?|rpt|lamme)'
    }

    def count_theories(abstract: str) -> List[str]:
        if pd.isna(abstract):
            return []
        abstract_lower = abstract.lower()
        return [t for t, p in theory_patterns.items() if re.search(p, abstract_lower)]

    print("\n📊 Analyzing cross-theory mentions...")

    df['theories_mentioned'] = df['Abstract'].apply(count_theories)
    df['n_theories'] = df['theories_mentioned'].apply(len)
    df['is_adversarial'] = df['n_theories'] >= 2

    yearly = df.groupby('Year').agg(
        total=('Title', 'count'),
        adversarial=('is_adversarial', 'sum')
    ).reset_index()
    yearly['pct'] = yearly['adversarial'] / yearly['total'] * 100
    yearly = yearly[(yearly['Year'] >= min_year) & (yearly['Year'] <= max_year)]

    print(f"\nTotal adversarial papers: {df['is_adversarial'].sum()}")
    print(f"Overall rate: {df['is_adversarial'].mean()*100:.1f}%")

    examples = df[df['is_adversarial']].head(3)
    print("\nExample adversarial papers:")
    for _, r in examples.iterrows():
        print(f"  - {r['Title'][:60]}... ({r['theories_mentioned']})")

    print("\n🎨 Creating visualization...")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly['Year'], y=yearly['pct'], name='Adversarial (%)', marker_color='#E63946'))

    z = np.polyfit(yearly['Year'], yearly['pct'], 1)
    trend_dir = "increasing" if z[0] > 0 else "decreasing"
    fig.add_trace(go.Scatter(x=yearly['Year'], y=np.poly1d(z)(yearly['Year']),
                             mode='lines', name='Trend', line=dict(color='#1D3557', width=2, dash='dash')))

    fig.update_layout(
        title=dict(text=f"<b>Adversarial Index: Cross-Theory Engagement</b><br><sup>Papers mentioning 2+ theories | Trend: {trend_dir}</sup>", font=dict(size=18)),
        xaxis=dict(title="Year"), yaxis=dict(title="% of Papers"),
        height=500, width=900
    )

    trend_color = "#2A9D8F" if z[0] > 0 else "#E76F51"
    fig.add_annotation(
        text=f"{'📈 Less siloed' if z[0] > 0 else '📉 More siloed'}",
        xref="paper", yref="paper", x=0.02, y=0.95, showarrow=False,
        font=dict(size=12, color=trend_color), bgcolor="white", borderpad=4
    )

    return fig, {'yearly': yearly.to_dict('records'), 'total_adversarial': int(df['is_adversarial'].sum()),
                 'rate': float(df['is_adversarial'].mean()), 'trend': trend_dir}

# =============================================================================
# CELL 11: SAVE RESULTS TO GOOGLE DRIVE
# =============================================================================

def save_results_to_drive(
    figures: Dict[str, go.Figure],
    df: pd.DataFrame,
    results: Dict[str, Any],
    output_folder: str
) -> None:
    """Save all results to Google Drive."""
    import os

    print("\n" + "=" * 60)
    print("💾 SAVING RESULTS TO GOOGLE DRIVE")
    print("=" * 60)

    os.makedirs(output_folder, exist_ok=True)

    # Save figures as HTML
    for name, fig in figures.items():
        if fig is not None:
            path = os.path.join(output_folder, f"{name}.html")
            fig.write_html(path)
            print(f"  ✅ Saved: {path}")

    # Save classified data
    csv_path = os.path.join(output_folder, "classified_papers.csv")
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Saved: {csv_path}")

    # Save summary JSON
    json_path = os.path.join(output_folder, "analysis_summary.json")
    with open(json_path, 'w') as f:
        # Convert numpy types to native Python types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        summary = {k: {k2: convert(v2) for k2, v2 in v.items()} if isinstance(v, dict) else convert(v)
                   for k, v in results.items()}
        json.dump(summary, f, indent=2, default=str)
    print(f"  ✅ Saved: {json_path}")

    print(f"\n📁 All results saved to: {output_folder}")

# =============================================================================
# CELL 12: MAIN EXECUTION PIPELINE
# =============================================================================

def run_full_audit(
    csv_path: str = CSV_PATH,
    api_key: str = GEMINI_API_KEY,
    output_folder: str = GDRIVE_OUTPUT_FOLDER,
    use_llm: bool = True,
    max_workers: int = MAX_WORKERS,
    save_to_drive: bool = True
) -> Dict[str, Any]:
    """
    Run the complete Consciousness Wars bibliometric audit.
    """
    print("\n" + "=" * 70)
    print("    🧠 CONSCIOUSNESS WARS BIBLIOMETRIC AUDIT 🧠")
    print("    IIT vs GNWT vs HOT vs RPT")
    print("=" * 70)

    # Load data
    df = load_bibliography(csv_path)

    # Extract classifications
    if use_llm:
        try:
            extractor = GeminiExtractor(api_key, GEMINI_MODEL)
            # Test connection
            test = extractor.extract_single("This paper investigates integrated information theory and phi.")
            print(f"✅ Gemini API connected | Test: {test['theory']}")
            df_classified = extractor.batch_extract_parallel(df, max_workers=max_workers)
        except Exception as e:
            print(f"⚠️ Gemini extraction failed: {e}")
            print("Falling back to rule-based extraction...")
            df_classified = RuleBasedExtractor().batch_extract(df)
    else:
        df_classified = RuleBasedExtractor().batch_extract(df)

    # Run analyses
    results = {}
    figures = {}

    # Part 1
    try:
        fig1, res1 = run_contrast_audit(df_classified)
        figures['contrast_sankey'] = fig1
        results['contrast_audit'] = res1
    except Exception as e:
        print(f"⚠️ ConTraSt audit failed: {e}")
        results['contrast_audit'] = {'error': str(e)}

    # Part 2
    try:
        fig2, res2 = run_moving_goalposts(df_classified, BASELINE_YEAR, MIN_YEAR, MAX_YEAR)
        figures['moving_goalposts'] = fig2
        results['moving_goalposts'] = res2
    except Exception as e:
        print(f"⚠️ Moving Goalposts failed: {e}")
        results['moving_goalposts'] = {'error': str(e)}

    # Part 3
    try:
        fig3, res3 = run_adversarial_index(df_classified, MIN_YEAR, MAX_YEAR)
        figures['adversarial_index'] = fig3
        results['adversarial_index'] = res3
    except Exception as e:
        print(f"⚠️ Adversarial Index failed: {e}")
        results['adversarial_index'] = {'error': str(e)}

    # Save to Drive
    if save_to_drive:
        try:
            save_results_to_drive(figures, df_classified, results, output_folder)
        except Exception as e:
            print(f"⚠️ Failed to save to Drive: {e}")
            # Save locally instead
            save_results_to_drive(figures, df_classified, results, "./outputs")

    # Summary
    print("\n" + "=" * 70)
    print("    ✅ AUDIT COMPLETE")
    print("=" * 70)
    print(f"\n📊 Papers analyzed: {len(df_classified)}")
    print(f"\n📈 Theory distribution:")
    for t, c in df_classified['theory'].value_counts().items():
        print(f"   {t}: {c} ({c/len(df_classified)*100:.1f}%)")

    return {'data': df_classified, 'results': results, 'figures': figures}

# =============================================================================
# CELL 13: RUN THE AUDIT (EXECUTE THIS CELL)
# =============================================================================

# Step 1: Mount Google Drive (run once)
# mount_google_drive()
# setup_output_folder(GDRIVE_OUTPUT_FOLDER)

# Step 2: Run the full audit
# results = run_full_audit()

# Step 3: Display visualizations
# results['figures']['contrast_sankey'].show()
# results['figures']['moving_goalposts'].show()
# results['figures']['adversarial_index'].show()

# Access data:
# results['data']  # DataFrame with all classifications

if __name__ == "__main__":
    # Quick run for testing
    print("Run the audit by uncommenting the lines in CELL 13")
    print("\nQuick start:")
    print("1. Set GEMINI_API_KEY at the top of this file")
    print("2. Run: mount_google_drive()")
    print("3. Run: results = run_full_audit()")
