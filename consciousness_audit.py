"""
Consciousness Wars Bibliometric Audit Pipeline
===============================================
A comprehensive bibliometric analysis of consciousness theories:
IIT (Integrated Information Theory), GNWT (Global Neuronal Workspace Theory),
HOT (Higher-Order Theory), and RPT (Recurrent Processing Theory).

Three Analyses:
1. ConTraSt Audit - Replication of Yaron et al. (2022)
2. Moving Goalposts - Concept drift analysis over time
3. Adversarial Index - Cross-theory citation patterns

Author: Bibliometric Analysis Pipeline
For use in Google Colab with high-quality LLM inference
"""

# =============================================================================
# CELL 1: INSTALLATION & SETUP
# =============================================================================
# Run this cell first to install all required dependencies

import subprocess
import sys

def install_packages():
    """Install required packages for the analysis."""
    packages = [
        'pandas',
        'numpy',
        'plotly',
        'scikit-learn',
        'sentence-transformers',
        'requests',
        'tqdm',
        'kaleido',  # For static image export from plotly
    ]

    print("=" * 60)
    print("INSTALLING REQUIRED PACKAGES")
    print("=" * 60)

    for package in packages:
        print(f"\n📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

    print("\n✅ All packages installed successfully!")
    print("=" * 60)

# Uncomment the line below to install packages (run once)
# install_packages()

# =============================================================================
# CELL 2: IMPORTS & CONFIGURATION
# =============================================================================

import pandas as pd
import numpy as np
import json
import re
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, asdict

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import dendrogram, linkage

import requests
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Configuration for the analysis pipeline."""

    # Data paths
    csv_path: str = "cons_bib.csv"  # Path to bibliography CSV
    output_dir: str = "outputs"     # Directory for outputs

    # LLM Configuration (Choose one)
    # Option 1: Ollama (local)
    llm_provider: str = "ollama"  # "ollama" or "gemini"
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3.1:8b"  # or "mistral", "phi3", etc.

    # Option 2: Google Gemini (cloud)
    gemini_api_key: str = ""  # Set your API key here
    gemini_model: str = "gemini-1.5-flash"

    # Analysis parameters
    theories: List[str] = None
    baseline_year: int = 2005
    min_year: int = 2000
    max_year: int = 2025

    # Rate limiting
    api_delay: float = 0.5  # Seconds between API calls
    max_retries: int = 3

    def __post_init__(self):
        if self.theories is None:
            self.theories = ["IIT", "GNWT", "HOT", "RPT", "Neutral"]

# Initialize configuration
CONFIG = Config()

# =============================================================================
# CELL 3: DATA LOADING & PREPROCESSING
# =============================================================================

def load_bibliography(csv_path: str) -> pd.DataFrame:
    """
    Load bibliography data from CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        DataFrame with bibliography data
    """
    print("\n" + "=" * 60)
    print("LOADING BIBLIOGRAPHY DATA")
    print("=" * 60)

    try:
        # Load CSV with proper encoding handling
        df = pd.read_csv(csv_path, encoding='utf-8-sig')

        # Standardize column names (remove BOM and whitespace)
        df.columns = df.columns.str.strip()

        print(f"✅ Loaded {len(df)} papers")
        print(f"📅 Year range: {df['Year'].min()} - {df['Year'].max()}")

        # Check for required columns
        required_cols = ['Abstract', 'Year', 'Title']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Filter out papers with empty abstracts
        df_valid = df[df['Abstract'].notna() & (df['Abstract'].str.len() > 50)].copy()
        print(f"📄 Papers with valid abstracts: {len(df_valid)}")

        # Convert year to integer
        df_valid['Year'] = pd.to_numeric(df_valid['Year'], errors='coerce')
        df_valid = df_valid[df_valid['Year'].notna()]
        df_valid['Year'] = df_valid['Year'].astype(int)

        return df_valid

    except FileNotFoundError:
        print(f"❌ Error: File not found at {csv_path}")
        print("Please upload your cons_bib.csv file or update the path.")
        raise
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def extract_first_sentences(text: str, n_sentences: int = 2) -> str:
    """Extract the first n sentences from text."""
    if pd.isna(text) or not text:
        return ""

    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return ' '.join(sentences[:n_sentences])

# =============================================================================
# CELL 4: LLM EXTRACTION MODULE
# =============================================================================

class LLMExtractor:
    """
    Extract structured information from abstracts using LLM.
    Supports both Ollama (local) and Gemini (cloud) backends.
    """

    EXTRACTION_PROMPT = """You are a consciousness science expert analyzing research papers.
Analyze this abstract and extract the following classifications:

ABSTRACT:
{abstract}

Classify this paper into the following categories. Respond ONLY with a valid JSON object:

{{
    "theory": "<IIT|GNWT|HOT|RPT|Neutral>",
    "paradigm": "<Report|No-Report>",
    "type": "<Content|State>",
    "epistemic": "<A Priori|Post-Hoc>",
    "confidence": <0.0-1.0>
}}

CLASSIFICATION GUIDELINES:

THEORY (which theory does the paper primarily support/investigate?):
- IIT: Integrated Information Theory (Tononi) - mentions phi, integration, information geometry, exclusion, intrinsic causal power
- GNWT: Global Neuronal Workspace Theory (Dehaene/Baars) - mentions global workspace, ignition, broadcasting, frontal-parietal
- HOT: Higher-Order Theory (Rosenthal/Lau) - mentions higher-order representations, metacognition, prefrontal
- RPT: Recurrent Processing Theory (Lamme) - mentions recurrent processing, feedback loops, V1, local recurrence
- Neutral: Does not clearly align with any specific theory, or is methodological/review

PARADIGM (experimental approach):
- Report: Participants verbally report or behaviorally indicate their conscious experience
- No-Report: Infers consciousness without requiring explicit reports (e.g., physiological measures only)

TYPE (what aspect of consciousness is studied?):
- Content: Studies WHAT we are conscious of (specific perceptions, thoughts)
- State: Studies WHETHER/HOW MUCH consciousness is present (levels, states)

EPISTEMIC (hypothesis approach):
- A Priori: Predictions derived from theory before data collection
- Post-Hoc: Analyses conducted after seeing the data, exploratory

Return ONLY the JSON object, no other text."""

    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()

    def extract_with_ollama(self, abstract: str) -> Dict[str, Any]:
        """Extract using local Ollama instance."""
        prompt = self.EXTRACTION_PROMPT.format(abstract=abstract[:2000])

        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 200
            }
        }

        try:
            response = self.session.post(
                self.config.ollama_url,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            text = result.get('response', '')

            # Extract JSON from response
            return self._parse_json_response(text)

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Ollama request failed: {e}")
            return self._default_extraction()

    def extract_with_gemini(self, abstract: str) -> Dict[str, Any]:
        """Extract using Google Gemini API."""
        if not self.config.gemini_api_key:
            raise ValueError("Gemini API key not configured")

        prompt = self.EXTRACTION_PROMPT.format(abstract=abstract[:2000])

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.gemini_model}:generateContent"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 200
            }
        }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.config.gemini_api_key
        }

        try:
            response = self.session.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()

            result = response.json()
            text = result['candidates'][0]['content']['parts'][0]['text']

            return self._parse_json_response(text)

        except Exception as e:
            print(f"⚠️ Gemini request failed: {e}")
            return self._default_extraction()

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                # Validate and normalize
                return {
                    "theory": self._validate_theory(data.get("theory", "Neutral")),
                    "paradigm": self._validate_paradigm(data.get("paradigm", "Report")),
                    "type": self._validate_type(data.get("type", "Content")),
                    "epistemic": self._validate_epistemic(data.get("epistemic", "Post-Hoc")),
                    "confidence": float(data.get("confidence", 0.5))
                }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            pass

        return self._default_extraction()

    def _validate_theory(self, value: str) -> str:
        valid = ["IIT", "GNWT", "HOT", "RPT", "Neutral"]
        return value if value in valid else "Neutral"

    def _validate_paradigm(self, value: str) -> str:
        valid = ["Report", "No-Report"]
        return value if value in valid else "Report"

    def _validate_type(self, value: str) -> str:
        valid = ["Content", "State"]
        return value if value in valid else "Content"

    def _validate_epistemic(self, value: str) -> str:
        valid = ["A Priori", "Post-Hoc"]
        return value if value in valid else "Post-Hoc"

    def _default_extraction(self) -> Dict[str, Any]:
        """Return default extraction when LLM fails."""
        return {
            "theory": "Neutral",
            "paradigm": "Report",
            "type": "Content",
            "epistemic": "Post-Hoc",
            "confidence": 0.0
        }

    def extract(self, abstract: str) -> Dict[str, Any]:
        """Extract information using configured LLM provider."""
        if self.config.llm_provider == "gemini":
            return self.extract_with_gemini(abstract)
        else:
            return self.extract_with_ollama(abstract)

    def batch_extract(self, df: pd.DataFrame, abstract_col: str = 'Abstract') -> pd.DataFrame:
        """Extract information from all abstracts in DataFrame."""
        print("\n" + "=" * 60)
        print("EXTRACTING PAPER CLASSIFICATIONS WITH LLM")
        print("=" * 60)
        print(f"Provider: {self.config.llm_provider}")
        print(f"Papers to process: {len(df)}")

        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
            abstract = row[abstract_col]

            if pd.isna(abstract) or len(str(abstract)) < 50:
                results.append(self._default_extraction())
                continue

            # Extract with retry logic
            for attempt in range(self.config.max_retries):
                try:
                    extraction = self.extract(abstract)
                    results.append(extraction)
                    break
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        results.append(self._default_extraction())
                    time.sleep(self.config.api_delay * (attempt + 1))

            time.sleep(self.config.api_delay)

        # Add results to DataFrame
        result_df = df.copy()
        for key in ['theory', 'paradigm', 'type', 'epistemic', 'confidence']:
            result_df[key] = [r[key] for r in results]

        print(f"\n✅ Extraction complete!")
        print(f"Theory distribution:")
        print(result_df['theory'].value_counts())

        return result_df

# =============================================================================
# CELL 5: RULE-BASED EXTRACTION (FALLBACK)
# =============================================================================

class RuleBasedExtractor:
    """
    Rule-based extraction as fallback when LLM is unavailable.
    Uses keyword matching for theory classification.
    """

    THEORY_KEYWORDS = {
        'IIT': [
            'integrated information', 'phi', 'tononi', 'information integration',
            'exclusion', 'intrinsic', 'causal power', 'iit', 'qualia space',
            'integrated information theory', 'compositional', 'intrinsic cause-effect'
        ],
        'GNWT': [
            'global workspace', 'global neuronal workspace', 'dehaene', 'baars',
            'ignition', 'broadcasting', 'access consciousness', 'gnwt', 'gnw',
            'frontal-parietal', 'frontoparietal', 'prefrontal broadcasting'
        ],
        'HOT': [
            'higher-order', 'higher order', 'rosenthal', 'metacognition',
            'hot theory', 'higher-order thought', 'higher-order representation',
            'lau', 'prefrontal', 'hot', 'meta-awareness'
        ],
        'RPT': [
            'recurrent processing', 'lamme', 'feedback', 'recurrent',
            'local recurrence', 'feedforward', 'v1', 'visual cortex recurrence',
            'rpt', 'recurrent loops'
        ]
    }

    PARADIGM_KEYWORDS = {
        'Report': [
            'report', 'verbal', 'button press', 'response', 'subjective',
            'asked', 'indicated', 'judgment', 'rating', 'self-report'
        ],
        'No-Report': [
            'no-report', 'no report', 'physiological', 'neural correlates only',
            'passive', 'inattentional', 'without report', 'implicit'
        ]
    }

    TYPE_KEYWORDS = {
        'Content': [
            'perception', 'visual', 'auditory', 'content', 'object',
            'feature', 'stimulus', 'representation', 'what', 'quale'
        ],
        'State': [
            'level', 'state', 'arousal', 'wakefulness', 'anesthesia',
            'sleep', 'coma', 'disorders of consciousness', 'vegetative',
            'minimally conscious', 'awareness level'
        ]
    }

    def __init__(self):
        pass

    def _keyword_score(self, text: str, keywords: List[str]) -> int:
        """Count keyword matches in text."""
        text_lower = text.lower()
        return sum(1 for kw in keywords if kw.lower() in text_lower)

    def extract(self, abstract: str) -> Dict[str, Any]:
        """Extract classifications using keyword matching."""
        if pd.isna(abstract) or not abstract:
            return self._default()

        abstract_lower = abstract.lower()

        # Theory classification
        theory_scores = {
            theory: self._keyword_score(abstract, keywords)
            for theory, keywords in self.THEORY_KEYWORDS.items()
        }
        max_score = max(theory_scores.values())
        theory = max(theory_scores, key=theory_scores.get) if max_score > 0 else "Neutral"

        # Paradigm classification
        paradigm_scores = {
            paradigm: self._keyword_score(abstract, keywords)
            for paradigm, keywords in self.PARADIGM_KEYWORDS.items()
        }
        paradigm = max(paradigm_scores, key=paradigm_scores.get) if max(paradigm_scores.values()) > 0 else "Report"

        # Type classification
        type_scores = {
            t: self._keyword_score(abstract, keywords)
            for t, keywords in self.TYPE_KEYWORDS.items()
        }
        content_type = max(type_scores, key=type_scores.get) if max(type_scores.values()) > 0 else "Content"

        # Epistemic (harder to determine from keywords)
        epistemic = "A Priori" if any(kw in abstract_lower for kw in ['predict', 'hypothesis', 'theory-driven']) else "Post-Hoc"

        confidence = min(max_score / 3.0, 1.0) if max_score > 0 else 0.3

        return {
            "theory": theory,
            "paradigm": paradigm,
            "type": content_type,
            "epistemic": epistemic,
            "confidence": confidence
        }

    def _default(self) -> Dict[str, Any]:
        return {
            "theory": "Neutral",
            "paradigm": "Report",
            "type": "Content",
            "epistemic": "Post-Hoc",
            "confidence": 0.0
        }

    def batch_extract(self, df: pd.DataFrame, abstract_col: str = 'Abstract') -> pd.DataFrame:
        """Extract information from all abstracts using rules."""
        print("\n" + "=" * 60)
        print("EXTRACTING PAPER CLASSIFICATIONS (RULE-BASED)")
        print("=" * 60)
        print(f"Papers to process: {len(df)}")

        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
            results.append(self.extract(row[abstract_col]))

        result_df = df.copy()
        for key in ['theory', 'paradigm', 'type', 'epistemic', 'confidence']:
            result_df[key] = [r[key] for r in results]

        print(f"\n✅ Extraction complete!")
        print(f"Theory distribution:")
        print(result_df['theory'].value_counts())

        return result_df

# =============================================================================
# CELL 6: PART 1 - CONTRAST AUDIT
# =============================================================================

def run_contrast_audit(df: pd.DataFrame) -> Tuple[go.Figure, Dict[str, Any]]:
    """
    PART 1: ConTraSt Audit - Replication of Yaron et al. (2022)

    Goal: Determine if Methodology predicts Theory

    Returns:
        Sankey diagram and analysis results
    """
    print("\n" + "=" * 60)
    print("PART 1: CONTRAST AUDIT")
    print("Does Methodology Predict Theory?")
    print("=" * 60)

    # Filter to papers with valid theory assignments (non-Neutral)
    df_theories = df[df['theory'] != 'Neutral'].copy()
    print(f"\nPapers with theory assignment: {len(df_theories)}")

    # Encode features for RandomForest
    le_paradigm = LabelEncoder()
    le_type = LabelEncoder()
    le_theory = LabelEncoder()

    df_theories['paradigm_enc'] = le_paradigm.fit_transform(df_theories['paradigm'])
    df_theories['type_enc'] = le_type.fit_transform(df_theories['type'])
    df_theories['theory_enc'] = le_theory.fit_transform(df_theories['theory'])

    # Features and target
    X = df_theories[['paradigm_enc', 'type_enc']].values
    y = df_theories['theory_enc'].values

    # Train RandomForest with cross-validation
    print("\n📊 Training RandomForest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)

    # Cross-validation scores
    cv_scores = cross_val_score(rf, X, y, cv=5)
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Fit final model
    rf.fit(X, y)

    # Feature importance
    feature_names = ['Paradigm', 'Type']
    importances = rf.feature_importances_
    print(f"\nFeature Importances:")
    for name, imp in zip(feature_names, importances):
        print(f"  {name}: {imp:.3f}")

    # Generate predictions for analysis
    y_pred = rf.predict(X)

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(
        y, y_pred,
        target_names=le_theory.classes_,
        zero_division=0
    ))

    # Create Sankey Diagram: Paradigm -> Theory
    print("\n🎨 Creating Sankey Diagram...")

    # Count flows from Paradigm to Theory
    paradigm_theory_counts = df_theories.groupby(['paradigm', 'theory']).size().reset_index(name='count')

    # Define nodes
    paradigms = df_theories['paradigm'].unique().tolist()
    theories = df_theories['theory'].unique().tolist()

    node_labels = paradigms + theories
    node_colors = ['#2E86AB', '#A23B72'] + ['#F18F01', '#C73E1D', '#3B1F2B', '#95190C'][:len(theories)]

    # Create links
    sources = []
    targets = []
    values = []
    link_colors = []

    for _, row in paradigm_theory_counts.iterrows():
        source_idx = paradigms.index(row['paradigm'])
        target_idx = len(paradigms) + theories.index(row['theory'])
        sources.append(source_idx)
        targets.append(target_idx)
        values.append(row['count'])
        # Color based on source
        link_colors.append('rgba(46, 134, 171, 0.4)' if row['paradigm'] == 'Report' else 'rgba(162, 59, 114, 0.4)')

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors[:len(node_labels)]
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    )])

    fig.update_layout(
        title=dict(
            text="<b>ConTraSt Audit: Paradigm → Theory Flow</b><br>" +
                 f"<sup>RF Accuracy: {cv_scores.mean():.1%} | n={len(df_theories)} papers</sup>",
            font=dict(size=18)
        ),
        font=dict(size=12),
        height=500,
        width=800
    )

    results = {
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'feature_importances': dict(zip(feature_names, importances)),
        'n_papers': len(df_theories),
        'paradigm_theory_counts': paradigm_theory_counts.to_dict('records')
    }

    return fig, results

# =============================================================================
# CELL 7: PART 2 - MOVING GOALPOSTS ANALYSIS
# =============================================================================

def run_moving_goalposts(df: pd.DataFrame, config: Config) -> Tuple[go.Figure, Dict[str, Any]]:
    """
    PART 2: Moving Goalposts Analysis - Concept Drift Detection

    Goal: Quantify semantic drift in theory definitions over time.

    Returns:
        Line chart of semantic drift and analysis results
    """
    print("\n" + "=" * 60)
    print("PART 2: MOVING GOALPOSTS ANALYSIS")
    print("Quantifying Concept Drift Over Time")
    print("=" * 60)

    # Load sentence transformer
    print("\n🤖 Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Filter to papers with specific theories
    df_theories = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])].copy()
    print(f"Papers with theory assignment: {len(df_theories)}")

    # Extract first 2 sentences (proxy for definition)
    df_theories['definition_text'] = df_theories['Abstract'].apply(
        lambda x: extract_first_sentences(x, n_sentences=2)
    )

    # Filter valid definitions
    df_theories = df_theories[df_theories['definition_text'].str.len() > 20]

    # Group by theory and year
    print("\n📊 Computing embeddings by theory and year...")

    drift_data = defaultdict(list)

    for theory in tqdm(['IIT', 'GNWT', 'HOT', 'RPT'], desc="Processing theories"):
        theory_df = df_theories[df_theories['theory'] == theory]

        if len(theory_df) < 5:
            print(f"  ⚠️ Skipping {theory}: insufficient papers ({len(theory_df)})")
            continue

        # Get all embeddings for this theory
        embeddings_by_year = {}

        for year in range(config.min_year, config.max_year + 1):
            year_df = theory_df[theory_df['Year'] == year]
            if len(year_df) >= 1:
                texts = year_df['definition_text'].tolist()
                embeddings = model.encode(texts)
                embeddings_by_year[year] = embeddings

        if not embeddings_by_year:
            continue

        # Find baseline year (earliest available or 2005)
        available_years = sorted(embeddings_by_year.keys())
        baseline_year = config.baseline_year if config.baseline_year in available_years else available_years[0]

        # Compute baseline centroid
        baseline_embeddings = embeddings_by_year[baseline_year]
        baseline_centroid = np.mean(baseline_embeddings, axis=0)

        # Compute drift for each year
        for year in available_years:
            year_embeddings = embeddings_by_year[year]
            year_centroid = np.mean(year_embeddings, axis=0)

            # Cosine distance from baseline
            distance = cosine(baseline_centroid, year_centroid)

            drift_data[theory].append({
                'year': year,
                'drift': distance,
                'n_papers': len(year_embeddings)
            })

    # Create line chart
    print("\n🎨 Creating drift visualization...")

    fig = go.Figure()

    colors = {
        'IIT': '#E63946',
        'GNWT': '#457B9D',
        'HOT': '#2A9D8F',
        'RPT': '#E9C46A'
    }

    for theory in ['IIT', 'GNWT', 'HOT', 'RPT']:
        if theory not in drift_data or not drift_data[theory]:
            continue

        theory_data = pd.DataFrame(drift_data[theory])

        # Smooth the data with rolling average
        if len(theory_data) > 3:
            theory_data = theory_data.sort_values('year')
            theory_data['drift_smooth'] = theory_data['drift'].rolling(window=3, min_periods=1, center=True).mean()
        else:
            theory_data['drift_smooth'] = theory_data['drift']

        fig.add_trace(go.Scatter(
            x=theory_data['year'],
            y=theory_data['drift_smooth'],
            mode='lines+markers',
            name=theory,
            line=dict(color=colors[theory], width=3),
            marker=dict(size=8),
            hovertemplate=(
                f"<b>{theory}</b><br>" +
                "Year: %{x}<br>" +
                "Semantic Drift: %{y:.3f}<br>" +
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title=dict(
            text="<b>Moving Goalposts: Semantic Drift in Theory Definitions</b><br>" +
                 "<sup>Cosine distance from baseline centroid (higher = more change)</sup>",
            font=dict(size=18)
        ),
        xaxis=dict(
            title="Year",
            tickmode='linear',
            tick0=config.min_year,
            dtick=5
        ),
        yaxis=dict(
            title="Semantic Drift (Cosine Distance)",
            rangemode='tozero'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        width=900,
        hovermode='x unified'
    )

    # Add annotation
    fig.add_annotation(
        text="↑ Rising line = Changing definition",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=11, color="gray"),
        align="left"
    )

    results = {
        'drift_data': {k: v for k, v in drift_data.items()},
        'baseline_year': config.baseline_year
    }

    return fig, results

# =============================================================================
# CELL 8: PART 3 - ADVERSARIAL INDEX
# =============================================================================

def run_adversarial_index(df: pd.DataFrame, config: Config) -> Tuple[go.Figure, Dict[str, Any]]:
    """
    PART 3: Adversarial Index - Cross-Theory Engagement

    Goal: Check if the field is becoming less siloed.

    Returns:
        Bar chart of adversarial papers over time
    """
    print("\n" + "=" * 60)
    print("PART 3: ADVERSARIAL INDEX")
    print("Measuring Cross-Theory Engagement")
    print("=" * 60)

    # Theory keywords for detection
    theory_patterns = {
        'IIT': r'\b(integrated information|iit|tononi|phi|information integration)\b',
        'GNWT': r'\b(global (neuronal )?workspace|gnw(t)?|dehaene|baars|ignition)\b',
        'HOT': r'\b(higher[- ]order|hot theory|rosenthal|metacognition)\b',
        'RPT': r'\b(recurrent processing|rpt|lamme|recurrent feedback)\b'
    }

    def count_theories_mentioned(abstract: str) -> List[str]:
        """Count how many rival theories are mentioned in abstract."""
        if pd.isna(abstract):
            return []

        abstract_lower = abstract.lower()
        mentioned = []

        for theory, pattern in theory_patterns.items():
            if re.search(pattern, abstract_lower):
                mentioned.append(theory)

        return mentioned

    # Analyze each paper
    print("\n📊 Analyzing cross-theory mentions...")

    df['theories_mentioned'] = df['Abstract'].apply(count_theories_mentioned)
    df['n_theories'] = df['theories_mentioned'].apply(len)
    df['is_adversarial'] = df['n_theories'] >= 2

    # Aggregate by year
    yearly_stats = df.groupby('Year').agg(
        total_papers=('Title', 'count'),
        adversarial_papers=('is_adversarial', 'sum')
    ).reset_index()

    yearly_stats['adversarial_pct'] = (
        yearly_stats['adversarial_papers'] / yearly_stats['total_papers'] * 100
    )

    # Filter to relevant year range
    yearly_stats = yearly_stats[
        (yearly_stats['Year'] >= config.min_year) &
        (yearly_stats['Year'] <= config.max_year)
    ]

    print(f"\nTotal adversarial papers: {df['is_adversarial'].sum()}")
    print(f"Overall adversarial rate: {df['is_adversarial'].mean()*100:.1f}%")

    # Show some examples
    adversarial_examples = df[df['is_adversarial']].head(3)
    print(f"\nExample adversarial papers:")
    for _, row in adversarial_examples.iterrows():
        print(f"  - {row['Title'][:60]}... ({row['theories_mentioned']})")

    # Create bar chart
    print("\n🎨 Creating visualization...")

    fig = go.Figure()

    # Primary bar: Adversarial percentage
    fig.add_trace(go.Bar(
        x=yearly_stats['Year'],
        y=yearly_stats['adversarial_pct'],
        name='Adversarial Papers (%)',
        marker_color='#E63946',
        hovertemplate=(
            "<b>Year: %{x}</b><br>" +
            "Adversarial: %{y:.1f}%<br>" +
            "<extra></extra>"
        )
    ))

    # Add trend line
    z = np.polyfit(yearly_stats['Year'], yearly_stats['adversarial_pct'], 1)
    p = np.poly1d(z)
    trend_y = p(yearly_stats['Year'])

    fig.add_trace(go.Scatter(
        x=yearly_stats['Year'],
        y=trend_y,
        mode='lines',
        name='Trend',
        line=dict(color='#1D3557', width=2, dash='dash'),
        hoverinfo='skip'
    ))

    # Determine trend direction
    trend_direction = "increasing" if z[0] > 0 else "decreasing"
    trend_color = "#2A9D8F" if z[0] > 0 else "#E76F51"

    fig.update_layout(
        title=dict(
            text="<b>Adversarial Index: Cross-Theory Engagement Over Time</b><br>" +
                 f"<sup>Papers mentioning 2+ rival theories | Trend: {trend_direction}</sup>",
            font=dict(size=18)
        ),
        xaxis=dict(
            title="Year",
            tickmode='linear',
            tick0=config.min_year,
            dtick=5
        ),
        yaxis=dict(
            title="Percentage of Papers (%)",
            rangemode='tozero'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        width=900,
        bargap=0.3
    )

    # Add annotation about interpretation
    fig.add_annotation(
        text=f"{'📈 Field becoming LESS siloed' if z[0] > 0 else '📉 Field becoming MORE siloed'}",
        xref="paper", yref="paper",
        x=0.02, y=0.95,
        showarrow=False,
        font=dict(size=12, color=trend_color, weight='bold'),
        align="left",
        bgcolor="white",
        borderpad=4
    )

    results = {
        'yearly_stats': yearly_stats.to_dict('records'),
        'total_adversarial': int(df['is_adversarial'].sum()),
        'overall_rate': float(df['is_adversarial'].mean()),
        'trend_slope': float(z[0]),
        'trend_direction': trend_direction
    }

    return fig, results

# =============================================================================
# CELL 9: MAIN EXECUTION PIPELINE
# =============================================================================

def run_full_audit(
    csv_path: str = "cons_bib.csv",
    use_llm: bool = True,
    llm_provider: str = "ollama",
    gemini_api_key: str = "",
    save_outputs: bool = True
) -> Dict[str, Any]:
    """
    Run the complete Consciousness Wars bibliometric audit.

    Args:
        csv_path: Path to bibliography CSV file
        use_llm: Whether to use LLM extraction (True) or rule-based (False)
        llm_provider: "ollama" or "gemini"
        gemini_api_key: API key for Gemini (if using)
        save_outputs: Whether to save HTML visualizations

    Returns:
        Dictionary containing all analysis results and figures
    """
    print("\n" + "=" * 70)
    print("    CONSCIOUSNESS WARS BIBLIOMETRIC AUDIT")
    print("    IIT vs GNWT vs HOT vs RPT")
    print("=" * 70)

    # Update configuration
    CONFIG.csv_path = csv_path
    CONFIG.llm_provider = llm_provider
    CONFIG.gemini_api_key = gemini_api_key

    # Step 1: Load data
    df = load_bibliography(csv_path)

    # Step 2: Extract paper classifications
    if use_llm:
        try:
            extractor = LLMExtractor(CONFIG)
            # Test connection
            test_result = extractor.extract("This is a test abstract about consciousness.")
            print(f"✅ LLM connection successful")
            df_classified = extractor.batch_extract(df)
        except Exception as e:
            print(f"⚠️ LLM extraction failed: {e}")
            print("Falling back to rule-based extraction...")
            extractor = RuleBasedExtractor()
            df_classified = extractor.batch_extract(df)
    else:
        print("Using rule-based extraction (LLM disabled)...")
        extractor = RuleBasedExtractor()
        df_classified = extractor.batch_extract(df)

    # Step 3: Run analyses
    results = {}
    figures = {}

    # Part 1: ConTraSt Audit
    try:
        fig1, results1 = run_contrast_audit(df_classified)
        figures['sankey'] = fig1
        results['contrast_audit'] = results1
        if save_outputs:
            fig1.write_html("contrast_sankey.html")
            print("💾 Saved: contrast_sankey.html")
    except Exception as e:
        print(f"⚠️ ConTraSt audit failed: {e}")
        results['contrast_audit'] = {'error': str(e)}

    # Part 2: Moving Goalposts
    try:
        fig2, results2 = run_moving_goalposts(df_classified, CONFIG)
        figures['drift'] = fig2
        results['moving_goalposts'] = results2
        if save_outputs:
            fig2.write_html("moving_goalposts.html")
            print("💾 Saved: moving_goalposts.html")
    except Exception as e:
        print(f"⚠️ Moving Goalposts analysis failed: {e}")
        results['moving_goalposts'] = {'error': str(e)}

    # Part 3: Adversarial Index
    try:
        fig3, results3 = run_adversarial_index(df_classified, CONFIG)
        figures['adversarial'] = fig3
        results['adversarial_index'] = results3
        if save_outputs:
            fig3.write_html("adversarial_index.html")
            print("💾 Saved: adversarial_index.html")
    except Exception as e:
        print(f"⚠️ Adversarial Index analysis failed: {e}")
        results['adversarial_index'] = {'error': str(e)}

    # Save classified data
    if save_outputs:
        df_classified.to_csv("classified_papers.csv", index=False)
        print("💾 Saved: classified_papers.csv")

    # Summary
    print("\n" + "=" * 70)
    print("    AUDIT COMPLETE")
    print("=" * 70)
    print(f"\n📊 Papers analyzed: {len(df_classified)}")
    print(f"📈 Theory distribution:")
    print(df_classified['theory'].value_counts().to_string())

    if 'contrast_audit' in results and 'cv_accuracy_mean' in results['contrast_audit']:
        print(f"\n🎯 ConTraSt Audit: Paradigm predicts theory with {results['contrast_audit']['cv_accuracy_mean']:.1%} accuracy")

    if 'adversarial_index' in results and 'trend_direction' in results['adversarial_index']:
        print(f"🔄 Adversarial Index: Field is {results['adversarial_index']['trend_direction']}")

    return {
        'data': df_classified,
        'results': results,
        'figures': figures
    }

# =============================================================================
# CELL 10: QUICK START (RUN THIS CELL)
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # CONFIGURATION - MODIFY THESE SETTINGS
    # =========================================================================

    # Path to your bibliography file
    CSV_PATH = "cons_bib.csv"

    # LLM Settings
    USE_LLM = True  # Set to False for rule-based extraction (no LLM needed)
    LLM_PROVIDER = "ollama"  # "ollama" or "gemini"

    # For Gemini, set your API key:
    # GEMINI_API_KEY = "your-api-key-here"
    GEMINI_API_KEY = ""

    # =========================================================================
    # RUN THE AUDIT
    # =========================================================================

    # For Google Colab with Gemini:
    # results = run_full_audit(
    #     csv_path=CSV_PATH,
    #     use_llm=True,
    #     llm_provider="gemini",
    #     gemini_api_key=GEMINI_API_KEY
    # )

    # For local with Ollama:
    # results = run_full_audit(
    #     csv_path=CSV_PATH,
    #     use_llm=True,
    #     llm_provider="ollama"
    # )

    # For rule-based (no LLM required):
    results = run_full_audit(
        csv_path=CSV_PATH,
        use_llm=False  # Uses keyword matching instead of LLM
    )

    # Access results:
    # results['figures']['sankey'].show()  # Sankey diagram
    # results['figures']['drift'].show()   # Semantic drift chart
    # results['figures']['adversarial'].show()  # Adversarial index chart
    # results['data']  # Classified DataFrame

# =============================================================================
# CELL 11: DISPLAY VISUALIZATIONS (COLAB)
# =============================================================================

def display_all_figures(results: Dict[str, Any]):
    """Display all figures in Colab/Jupyter environment."""
    if 'figures' in results:
        for name, fig in results['figures'].items():
            print(f"\n{'='*60}")
            print(f"  {name.upper()}")
            print('='*60)
            fig.show()

# Uncomment to display after running:
# display_all_figures(results)
