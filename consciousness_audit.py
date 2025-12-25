"""
================================================================================
CONSCIOUSNESS WARS BIBLIOMETRIC AUDIT PIPELINE v3.0
================================================================================
A comprehensive bibliometric analysis of consciousness theories:
IIT (Integrated Information Theory), GNWT (Global Neuronal Workspace Theory),
HOT (Higher-Order Theory), and RPT (Recurrent Processing Theory).

ANALYSES:
1-7: Original analyses (ConTraSt, Moving Goalposts, Adversarial, etc.)
8: Semantic Space Clustering with UMAP visualization
9: Hypothesis Testing (Implicit Bias, Schism, Insularity)

KEY FEATURES:
- "Fingerprint" classification with theory-specific markers
- Gemini text-embedding-004 for semantic embeddings
- UMAP dimensionality reduction
- Statistical hypothesis testing

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

GEMINI_API_KEY = ""  # <-- PASTE YOUR API KEY HERE
OUTPUT_FOLDER = "./results"
CSV_PATH = "cons_bib.csv"
GEMINI_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "text-embedding-004"

MAX_WORKERS = 5
API_DELAY = 0.2
MAX_RETRIES = 3
EMBEDDING_BATCH_SIZE = 100

BASELINE_YEAR = 2005
MIN_YEAR = 2000
MAX_YEAR = 2025
SCHISM_YEAR = 2015

DEBUG_MODE = False
DEBUG_SAMPLE_SIZE = 5

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
# CELL 5: FINGERPRINT PROMPT
# =============================================================================

FINGERPRINT_PROMPT = '''You are an expert in consciousness science. Classify this abstract by detecting theoretical "fingerprints".

ABSTRACT:
{abstract}

THEORY FINGERPRINTS:

🔵 IIT Markers: "Phi" (Φ), "integrated information", "differentiation" + "integration", "cause-effect structure", "posterior hot zone", "maximally irreducible", "exclusion postulate", "qualia space". Authors: Tononi, Koch, Massimini, Boly

🟢 GNWT Markers: "Global workspace", "ignition", "P3b"/"P300", "broadcasting", "long-distance connectivity", "fronto-parietal", "access consciousness", "late cortical activity" (300-500ms). Authors: Dehaene, Baars, Changeux, Sergent

🟡 HOT Markers: "Higher-order thought", "higher-order representation", "meta-cognition", "awareness of awareness", "second-order", "introspection" as mechanism, "prefrontal" for awareness. Authors: Rosenthal, Lau, Brown, LeDoux

🔴 RPT Markers: "Local recurrence" in SENSORY areas, "re-entrant processing" in visual cortex, "feedback within V1/V2/V4", "feedforward vs feedback" in sensory processing. Authors: Lamme, Super, Fahrenfort
⚠️ NOT RPT: "feedback from PFC", "top-down attention from frontal areas", "fronto-parietal recurrence" (these are GNWT!)

🔘 NEUTRAL: No clear theory fingerprints, methodology paper, or equally supports multiple theories.

OTHER DIMENSIONS:
2. PARADIGM: Report / No-Report
3. TYPE: Content / State
4. EPISTEMIC: A Priori / Post-Hoc
5. ANATOMY: Top 3 brain regions ["PFC", "V1"] or ["None"]
6. TONE toward rivals: Dismissive / Critical / Constructive / Neutral
7. TARGET: Phenomenology / Function / Mechanism
8. SUBJECT: Human / Clinical / Animal / Simulation / Review

Respond ONLY with JSON:
{{"theory":"IIT","paradigm":"Report","type":"Content","epistemic":"A Priori","anatomy":["PFC"],"tone":"Neutral","target":"Function","subject":"Human","confidence":0.85,"fingerprints_found":["phi","posterior hot zone"]}}'''

# =============================================================================
# CELL 6: GEMINI EXTRACTOR
# =============================================================================

class GeminiExtractor:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key, self.model = api_key, model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.session = requests.Session()
        self.debug_responses = []
        if not api_key: raise ValueError("❌ GEMINI_API_KEY not set!")

    def _call_gemini(self, abstract: str) -> Tuple[Dict, str]:
        payload = {"contents": [{"parts": [{"text": FINGERPRINT_PROMPT.format(abstract=abstract[:2500])}]}],
                   "generationConfig": {"temperature": 0.0, "maxOutputTokens": 300}}
        try:
            r = self.session.post(f"{self.base_url}?key={self.api_key}", json=payload, timeout=30)
            if r.status_code == 429: time.sleep(3); return {"_retry": True}, "RATE_LIMITED"
            if r.status_code != 200: return self._default(), f"HTTP_{r.status_code}"
            result = r.json()
            if 'candidates' in result and result['candidates']:
                raw = result['candidates'][0]['content']['parts'][0].get('text', '')
                return self._parse(raw), raw
            return self._default(), "NO_CANDIDATES"
        except Exception as e: return self._default(), str(e)[:50]

    def _parse(self, text: str) -> Dict:
        text = re.sub(r'^```json\s*\n?|```\s*$', '', text.strip())
        try: return self._normalize(json.loads(text))
        except: pass
        m = re.search(r'\{[^{}]*"theory"[^{}]*\}', text, re.I | re.DOTALL)
        if m:
            try: return self._normalize(json.loads(m.group()))
            except: pass
        return self._default()

    def _normalize(self, d: Dict) -> Dict:
        r = self._default()
        t = str(d.get("theory", "")).upper()
        r["theory"] = t if t in ["IIT", "GNWT", "HOT", "RPT"] else "Neutral"
        r["paradigm"] = "No-Report" if "no" in str(d.get("paradigm", "")).lower() else "Report"
        r["type"] = "State" if "state" in str(d.get("type", "")).lower() else "Content"
        r["epistemic"] = "A Priori" if "priori" in str(d.get("epistemic", "")).lower() else "Post-Hoc"
        anat = d.get("anatomy", [])
        r["anatomy"] = [str(a).strip() for a in (anat if isinstance(anat, list) else anat.split(",")) if a and str(a).lower() != "none"][:3]
        tone = str(d.get("tone", "")).lower()
        r["tone"] = "Dismissive" if "dismiss" in tone else "Critical" if "critic" in tone else "Constructive" if "construct" in tone else "Neutral"
        tgt = str(d.get("target", "")).lower()
        r["target"] = "Phenomenology" if "phenom" in tgt else "Mechanism" if "mechan" in tgt else "Function"
        subj = str(d.get("subject", "")).lower()
        r["subject"] = "Clinical" if "clinical" in subj else "Animal" if "animal" in subj else "Simulation" if "simul" in subj else "Review" if "review" in subj else "Human"
        r["confidence"] = float(d.get("confidence", 0.7))
        r["fingerprints_found"] = d.get("fingerprints_found", [])
        return r

    def _default(self) -> Dict:
        return {"theory": "Neutral", "paradigm": "Report", "type": "Content", "epistemic": "Post-Hoc",
                "anatomy": [], "tone": "Neutral", "target": "Function", "subject": "Human",
                "confidence": 0.0, "fingerprints_found": []}

    def extract_single(self, abstract: str, idx: int = 0) -> Dict:
        if pd.isna(abstract) or len(str(abstract)) < 50: return self._default()
        for attempt in range(MAX_RETRIES):
            result, raw = self._call_gemini(str(abstract))
            if DEBUG_MODE and len(self.debug_responses) < DEBUG_SAMPLE_SIZE:
                self.debug_responses.append({"idx": idx, "raw": raw[:500], "parsed": result})
            if result.get("_retry"): time.sleep(2 * (attempt + 1)); continue
            return result
        return self._default()

    def batch_extract_parallel(self, df: pd.DataFrame, max_workers: int = 5) -> pd.DataFrame:
        print(f"\n{'='*60}\n🔍 FINGERPRINT CLASSIFICATION\n{'='*60}")
        print(f"Model: {self.model} | Papers: {len(df)} | Workers: {max_workers}")
        results = [None] * len(df)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.extract_single, row['Abstract'], i): i for i, row in df.iterrows()}
            for f in tqdm(as_completed(futures), total=len(df), desc="Classifying"):
                results[futures[f]] = f.result()
                time.sleep(API_DELAY)
        if DEBUG_MODE and self.debug_responses:
            print("\n🔍 DEBUG SAMPLES:")
            for d in self.debug_responses: print(f"  {d['idx']}: {d['parsed']['theory']}")
        df_out = df.copy()
        for k in ['theory', 'paradigm', 'type', 'epistemic', 'anatomy', 'tone', 'target', 'subject', 'confidence', 'fingerprints_found']:
            df_out[k] = [r.get(k, self._default()[k]) for r in results]
        print(f"\n📊 Theory distribution:")
        for t, c in df_out['theory'].value_counts().items():
            print(f"   {t}: {c} ({c/len(df_out)*100:.1f}%)")
        return df_out

# =============================================================================
# CELL 7: GEMINI EMBEDDER
# =============================================================================

class GeminiEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-004"):
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
                    reqs.append({"model": f"models/{self.model}", "content": {"parts": [{"text": str(t)[:2000]}]}})
                    idxs.append(start + i)
            if not reqs: continue
            try:
                r = self.session.post(f"{self.batch_url}?key={self.api_key}", json={"requests": reqs}, timeout=60)
                if r.status_code == 429: time.sleep(30); continue
                r.raise_for_status()
                for j, emb in enumerate(r.json().get('embeddings', [])):
                    if 'values' in emb: embeddings[idxs[j]] = np.array(emb['values'])
            except Exception as e: print(f"⚠️ Batch error: {e}")
            time.sleep(0.5)
        print(f"✅ Embedded: {sum(1 for e in embeddings if e is not None)}/{len(texts)}")
        return embeddings

# =============================================================================
# CELL 8: SEMANTIC CLUSTERING
# =============================================================================

def run_semantic_clustering(df: pd.DataFrame, embeddings: List, output_folder: str) -> Dict:
    print(f"\n{'='*60}\n🌌 SEMANTIC CLUSTERING\n{'='*60}")
    if not UMAP_AVAILABLE or not MATPLOTLIB_AVAILABLE: return {"error": "Missing dependencies"}
    valid_mask = [e is not None for e in embeddings]
    X = np.vstack([e for e in embeddings if e is not None])
    valid_df = df[valid_mask].copy()
    if len(X) < 50: return {"error": "Insufficient embeddings"}
    print(f"Valid: {len(X)} | Shape: {X.shape}")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    X_2d = reducer.fit_transform(X)
    valid_df['umap_x'], valid_df['umap_y'] = X_2d[:, 0], X_2d[:, 1]
    plt.figure(figsize=(14, 10))
    colors = {'IIT': '#E63946', 'GNWT': '#457B9D', 'HOT': '#2A9D8F', 'RPT': '#E9C46A', 'Neutral': '#CCCCCC'}
    for t in ['Neutral', 'IIT', 'GNWT', 'HOT', 'RPT']:
        m = valid_df['theory'] == t
        if m.sum(): plt.scatter(valid_df.loc[m, 'umap_x'], valid_df.loc[m, 'umap_y'], c=colors[t],
                                label=f"{t} (n={m.sum()})", alpha=0.6 if t == 'Neutral' else 0.8, s=30 if t == 'Neutral' else 50)
    plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2'); plt.title('Semantic Space by Theory'); plt.legend()
    plt.savefig(os.path.join(output_folder, "8_semantic_space.png"), dpi=300, bbox_inches='tight'); plt.close()
    print("  ✅ Saved: 8_semantic_space.png")
    centroids = {t: np.mean(X[valid_df['theory'] == t], axis=0) for t in ['IIT', 'GNWT', 'HOT', 'RPT', 'Neutral'] if (valid_df['theory'] == t).sum()}
    return {"valid_df": valid_df, "centroids": centroids, "X": X}

# =============================================================================
# CELL 9: HYPOTHESIS TESTING
# =============================================================================

def run_hypothesis_tests(df: pd.DataFrame, centroids: Dict, X: np.ndarray, output_folder: str) -> Dict:
    print(f"\n{'='*60}\n🔬 HYPOTHESIS TESTING\n{'='*60}")
    results = {}

    # A: Implicit Bias
    print("\n📊 HYPOTHESIS A: Implicit Bias")
    if 'IIT' in centroids and 'GNWT' in centroids:
        neutral = X[df['theory'] == 'Neutral']
        if len(neutral) > 10:
            d_iit = [euclidean(e, centroids['IIT']) for e in neutral]
            d_gnwt = [euclidean(e, centroids['GNWT']) for e in neutral]
            t, p = stats.ttest_rel(d_iit, d_gnwt)
            closer = "IIT" if np.mean(d_iit) < np.mean(d_gnwt) else "GNWT"
            print(f"  Neutral papers closer to: {closer} | t={t:.3f}, p={p:.4f} {'✅' if p < 0.05 else '❌'}")
            results['hyp_a'] = {'closer': closer, 'p': p, 'sig': p < 0.05}

    # B: Schism
    print("\n📊 HYPOTHESIS B: The Schism")
    if 'IIT' in centroids and 'GNWT' in centroids:
        pre, post = df[df['Year'] < SCHISM_YEAR], df[df['Year'] >= SCHISM_YEAR]
        def cdist(subset, t1, t2):
            m1, m2 = (df['theory'] == t1) & df.index.isin(subset.index), (df['theory'] == t2) & df.index.isin(subset.index)
            if m1.sum() < 3 or m2.sum() < 3: return None
            return euclidean(np.mean(X[m1], axis=0), np.mean(X[m2], axis=0))
        d_pre, d_post = cdist(pre, 'IIT', 'GNWT'), cdist(post, 'IIT', 'GNWT')
        if d_pre and d_post:
            direction = "DIVERGING" if d_post > d_pre else "CONVERGING"
            print(f"  Pre-2015: {d_pre:.4f} | Post-2015: {d_post:.4f} | {direction}")
            results['hyp_b'] = {'d_pre': d_pre, 'd_post': d_post, 'direction': direction}

    # C: Insularity
    print("\n📊 HYPOTHESIS C: Insularity")
    densities = {}
    for t in ['IIT', 'GNWT', 'HOT', 'RPT']:
        if t in centroids:
            te = X[df['theory'] == t]
            if len(te) >= 5:
                d = [euclidean(e, centroids[t]) for e in te]
                densities[t] = {'n': len(te), 'mean': np.mean(d), 'std': np.std(d)}
                print(f"  {t}: n={len(te)}, mean_dist={np.mean(d):.4f}")
    if densities:
        most = min(densities, key=lambda x: densities[x]['mean'])
        print(f"  Most insular: {most}")
        results['hyp_c'] = {'most_insular': most, 'densities': densities}
        if MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=(10, 6))
            plt.bar(densities.keys(), [densities[t]['mean'] for t in densities],
                   yerr=[densities[t]['std'] for t in densities], capsize=5,
                   color=['#E63946', '#457B9D', '#2A9D8F', '#E9C46A'][:len(densities)])
            plt.ylabel('Mean Distance to Centroid'); plt.title('Cluster Density (Lower = More Insular)')
            plt.savefig(os.path.join(output_folder, "9_cluster_density.png"), dpi=300); plt.close()
            print("  ✅ Saved: 9_cluster_density.png")
    return results

# =============================================================================
# CELL 10: RULE-BASED FALLBACK
# =============================================================================

class RuleBasedExtractor:
    FINGERPRINTS = {
        'IIT': ['integrated information', 'phi', 'tononi', 'cause-effect', 'posterior hot zone', 'maximally irreducible'],
        'GNWT': ['global workspace', 'dehaene', 'baars', 'ignition', 'broadcasting', 'p3b', 'fronto-parietal'],
        'HOT': ['higher-order', 'rosenthal', 'metacognition', 'awareness of awareness', 'lau'],
        'RPT': ['recurrent processing', 'lamme', 'local recurrence', 'feedforward sweep']
    }
    RPT_EXCLUSIONS = ['prefrontal feedback', 'top-down attention', 'global workspace']

    def extract(self, abstract: str) -> Dict:
        if pd.isna(abstract): return self._default()
        text = ' ' + abstract.lower() + ' '
        scores = {t: sum(1 for k in kws if k in text) for t, kws in self.FINGERPRINTS.items()}
        if scores.get('RPT', 0) > 0:
            for ex in self.RPT_EXCLUSIONS:
                if ex in text: scores['RPT'] = max(0, scores['RPT'] - 2)
        theory = max(scores, key=scores.get) if max(scores.values()) > 0 else "Neutral"
        return {"theory": theory, "paradigm": "Report", "type": "Content", "epistemic": "Post-Hoc",
                "anatomy": [], "tone": "Neutral", "target": "Function", "subject": "Human",
                "confidence": min(max(scores.values()) / 3, 1), "fingerprints_found": []}

    def _default(self): return {"theory": "Neutral", "paradigm": "Report", "type": "Content",
                                "epistemic": "Post-Hoc", "anatomy": [], "tone": "Neutral",
                                "target": "Function", "subject": "Human", "confidence": 0, "fingerprints_found": []}

    def batch_extract(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"\n{'='*60}\n📝 RULE-BASED EXTRACTION\n{'='*60}")
        results = [self.extract(row['Abstract']) for _, row in tqdm(df.iterrows(), total=len(df))]
        df_out = df.copy()
        for k in results[0].keys(): df_out[k] = [r[k] for r in results]
        for t, c in df_out['theory'].value_counts().items(): print(f"   {t}: {c}")
        return df_out

# =============================================================================
# CELL 11-17: ANALYSES 1-7
# =============================================================================

def run_contrast_audit(df, out):
    print(f"\n{'='*60}\n📊 ANALYSIS 1: CONTRAST\n{'='*60}")
    dft = df[df['theory'] != 'Neutral'].copy()
    if len(dft) < 20: return {"error": "Insufficient"}
    le_p, le_t, le_th = LabelEncoder(), LabelEncoder(), LabelEncoder()
    dft['p'], dft['t'], dft['th'] = le_p.fit_transform(dft['paradigm']), le_t.fit_transform(dft['type']), le_th.fit_transform(dft['theory'])
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = cross_val_score(rf, dft[['p', 't']], dft['th'], cv=min(5, len(set(dft['th']))))
    print(f"  CV Accuracy: {cv.mean():.3f}")
    rf.fit(dft[['p', 't']], dft['th'])
    cts = dft.groupby(['paradigm', 'theory']).size().reset_index(name='c')
    pars, ths = dft['paradigm'].unique().tolist(), dft['theory'].unique().tolist()
    fig = go.Figure(go.Sankey(node=dict(label=pars+ths), link=dict(
        source=[pars.index(r['paradigm']) for _, r in cts.iterrows()],
        target=[len(pars)+ths.index(r['theory']) for _, r in cts.iterrows()],
        value=cts['c'].tolist())))
    fig.update_layout(title=f"ConTraSt (Acc: {cv.mean():.1%})"); fig.write_image(os.path.join(out, "1_contrast.png"), scale=2)
    print("  ✅ Saved: 1_contrast.png")
    return {"acc": cv.mean()}

def run_moving_goalposts(df, out):
    print(f"\n{'='*60}\n📈 ANALYSIS 2: GOALPOSTS\n{'='*60}")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    dft = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])].copy()
    dft['txt'] = dft['Abstract'].apply(lambda x: extract_first_sentences(x, 2))
    colors = {'IIT': '#E63946', 'GNWT': '#457B9D', 'HOT': '#2A9D8F', 'RPT': '#E9C46A'}
    fig = go.Figure()
    for t in ['IIT', 'GNWT', 'HOT', 'RPT']:
        tdf = dft[dft['theory'] == t]
        if len(tdf) < 3: continue
        emb_yr = {y: model.encode(tdf[tdf['Year']==y]['txt'].tolist()) for y in range(MIN_YEAR, MAX_YEAR+1) if len(tdf[tdf['Year']==y]) > 0}
        if not emb_yr: continue
        yrs = sorted(emb_yr.keys()); base = BASELINE_YEAR if BASELINE_YEAR in yrs else yrs[0]
        drift = [{"y": y, "d": cosine(np.mean(emb_yr[base], 0), np.mean(emb_yr[y], 0))} for y in yrs]
        td = pd.DataFrame(drift)
        fig.add_trace(go.Scatter(x=td['y'], y=td['d'], mode='lines+markers', name=t, line=dict(color=colors[t])))
    fig.update_layout(title="Moving Goalposts", xaxis_title="Year", yaxis_title="Drift")
    fig.write_image(os.path.join(out, "2_goalposts.png"), scale=2)
    print("  ✅ Saved: 2_goalposts.png")
    return {}

def run_adversarial(df, out):
    print(f"\n{'='*60}\n⚔️ ANALYSIS 3: ADVERSARIAL\n{'='*60}")
    pats = {'IIT': r'\b(integrated information|iit\b|tononi)', 'GNWT': r'\b(global.?workspace|gnw|dehaene)',
            'HOT': r'\b(higher.?order|rosenthal)', 'RPT': r'\b(recurrent processing|lamme)'}
    df['adv'] = df['Abstract'].apply(lambda x: len([t for t, p in pats.items() if re.search(p, str(x).lower())]) >= 2)
    yearly = df.groupby('Year').agg(tot=('Title', 'count'), adv=('adv', 'sum')).reset_index()
    yearly['pct'] = yearly['adv'] / yearly['tot'] * 100
    yearly = yearly[(yearly['Year'] >= MIN_YEAR) & (yearly['Year'] <= MAX_YEAR)]
    z = np.polyfit(yearly['Year'], yearly['pct'], 1)
    fig = go.Figure([go.Bar(x=yearly['Year'], y=yearly['pct']), go.Scatter(x=yearly['Year'], y=np.poly1d(z)(yearly['Year']), mode='lines')])
    fig.update_layout(title=f"Adversarial Index (Trend: {'↑' if z[0]>0 else '↓'})")
    fig.write_image(os.path.join(out, "3_adversarial.png"), scale=2)
    print(f"  ✅ Saved: 3_adversarial.png | Total: {df['adv'].sum()}")
    return {"total": int(df['adv'].sum())}

def run_cartography(df, out):
    print(f"\n{'='*60}\n🧠 ANALYSIS 4: CARTOGRAPHY\n{'='*60}")
    dft = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])]
    if len(dft) < 10: return {"error": "Insufficient"}
    rc = defaultdict(lambda: defaultdict(int))
    for _, r in dft.iterrows():
        for a in (r.get('anatomy', []) or []):
            if a and str(a).lower() != 'none': rc[r['theory']][str(a).upper()] += 1
    allr = Counter(); [allr.update(v) for v in rc.values()]
    top = [r for r, _ in allr.most_common(15)]
    if not top: return {"error": "No regions"}
    mat = [[rc[t].get(r, 0) for r in top] for t in ['IIT', 'GNWT', 'HOT', 'RPT']]
    mat = [[x/max(sum(row),1)*100 for x in row] for row in mat]
    fig = go.Figure(go.Heatmap(z=mat, x=top, y=['IIT', 'GNWT', 'HOT', 'RPT'], colorscale='Reds'))
    fig.update_layout(title="Neuro-Cartography"); fig.write_image(os.path.join(out, "4_cartography.png"), scale=2)
    print("  ✅ Saved: 4_cartography.png")
    return {}

def run_aggression(df, out):
    print(f"\n{'='*60}\n😤 ANALYSIS 5: AGGRESSION\n{'='*60}")
    dft = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])]
    if len(dft) < 10: return {"error": "Insufficient"}
    yearly = dft.groupby('Year').apply(lambda x: ((x['tone']=='Dismissive')|(x['tone']=='Critical')).sum()/len(x)*100).reset_index(name='pct')
    yearly = yearly[(yearly['Year'] >= MIN_YEAR) & (yearly['Year'] <= MAX_YEAR)]
    if len(yearly) < 3: return {"error": "Insufficient"}
    z = np.polyfit(yearly['Year'], yearly['pct'], 1)
    fig = go.Figure([go.Scatter(x=yearly['Year'], y=yearly['pct'], mode='lines+markers'),
                     go.Scatter(x=yearly['Year'], y=np.poly1d(z)(yearly['Year']), mode='lines', line=dict(dash='dash'))])
    fig.update_layout(title=f"Aggression Index (Trend: {'↑' if z[0]>0 else '↓'})")
    fig.write_image(os.path.join(out, "5_aggression.png"), scale=2)
    print("  ✅ Saved: 5_aggression.png")
    return {}

def run_easy_hard(df, out):
    print(f"\n{'='*60}\n🤔 ANALYSIS 6: EASY VS HARD\n{'='*60}")
    dft = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])]
    if len(dft) < 10: return {"error": "Insufficient"}
    cts = dft.groupby(['theory', 'target']).size().unstack(fill_value=0)
    for c in ['Phenomenology', 'Function', 'Mechanism']:
        if c not in cts.columns: cts[c] = 0
    pct = cts.div(cts.sum(1), 0) * 100
    fig = go.Figure([go.Bar(x=pct.index, y=pct[t], name=t) for t in ['Phenomenology', 'Function', 'Mechanism']])
    fig.update_layout(title="Easy vs Hard", barmode='stack'); fig.write_image(os.path.join(out, "6_easy_hard.png"), scale=2)
    print("  ✅ Saved: 6_easy_hard.png")
    return {}

def run_organism(df, out):
    print(f"\n{'='*60}\n🐁 ANALYSIS 7: ORGANISM\n{'='*60}")
    dft = df[df['theory'].isin(['IIT', 'GNWT', 'HOT', 'RPT'])]
    if len(dft) < 10: return {"error": "Insufficient"}
    cts = dft.groupby(['theory', 'subject']).size().unstack(fill_value=0)
    pct = cts.div(cts.sum(1), 0) * 100
    fig = go.Figure([go.Bar(x=pct.index, y=pct.get(s, [0]*len(pct)), name=s) for s in ['Human', 'Clinical', 'Animal', 'Simulation', 'Review'] if s in pct.columns])
    fig.update_layout(title="Model Organism", barmode='stack'); fig.write_image(os.path.join(out, "7_organism.png"), scale=2)
    print("  ✅ Saved: 7_organism.png")
    return {}

# =============================================================================
# CELL 18: MAIN PIPELINE
# =============================================================================

def run_full_audit(csv_path=CSV_PATH, api_key=GEMINI_API_KEY, output_folder=OUTPUT_FOLDER,
                   use_llm=True, use_rules_only=False, run_embeddings=True):
    print(f"\n{'='*70}\n    🧠 CONSCIOUSNESS WARS AUDIT v3.0 🧠\n{'='*70}")
    os.makedirs(output_folder, exist_ok=True)
    df = load_bibliography(csv_path)

    if use_rules_only:
        df_c = RuleBasedExtractor().batch_extract(df)
    elif use_llm and api_key:
        try:
            ext = GeminiExtractor(api_key, GEMINI_MODEL)
            test = ext.extract_single("Integrated information theory and phi measures.", 0)
            print(f"✅ Gemini connected | Test: {test['theory']}")
            df_c = ext.batch_extract_parallel(df, MAX_WORKERS)
        except Exception as e:
            print(f"⚠️ Gemini failed: {e}\nFalling back to rules...")
            df_c = RuleBasedExtractor().batch_extract(df)
    else:
        df_c = RuleBasedExtractor().batch_extract(df)

    results = {}
    for name, fn in [('contrast', run_contrast_audit), ('goalposts', run_moving_goalposts),
                     ('adversarial', run_adversarial), ('cartography', run_cartography),
                     ('aggression', run_aggression), ('easy_hard', run_easy_hard), ('organism', run_organism)]:
        try: results[name] = fn(df_c, output_folder)
        except Exception as e: print(f"⚠️ {name} failed: {e}")

    if run_embeddings and api_key:
        try:
            embs = GeminiEmbedder(api_key, EMBEDDING_MODEL).embed_batch(df_c['Abstract'].tolist(), EMBEDDING_BATCH_SIZE)
            clust = run_semantic_clustering(df_c, embs, output_folder)
            if 'error' not in clust:
                results['hyp'] = run_hypothesis_tests(clust['valid_df'], clust['centroids'], clust['X'], output_folder)
        except Exception as e: print(f"⚠️ Clustering failed: {e}")

    df_c.to_csv(os.path.join(output_folder, "classified_papers.csv"), index=False)
    with open(os.path.join(output_folder, "results.json"), 'w') as f: json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}\n    ✅ AUDIT COMPLETE\n{'='*70}")
    print(f"Papers: {len(df_c)}")
    for t, c in df_c['theory'].value_counts().items(): print(f"   {t}: {c} ({c/len(df_c)*100:.1f}%)")
    print(f"📁 Results: {output_folder}")
    return {'data': df_c, 'results': results}

if __name__ == "__main__":
    print("CONSCIOUSNESS WARS AUDIT v3.0\n\n1. Set GEMINI_API_KEY\n2. Run: results = run_full_audit()")
