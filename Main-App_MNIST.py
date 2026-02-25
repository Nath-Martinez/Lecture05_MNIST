import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST Classifier Studio",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# THEME CONSTANTS
# ─────────────────────────────────────────────────────────────
BG     = "#0f1117"
BG2    = "#1a1d27"
MUTED  = "#8890b5"
LIGHT  = "#c5cae9"

DIGIT_COLORS = [
    "#FF6B6B","#FFD93D","#6BCB77","#4D96FF","#C77DFF",
    "#FF9A3C","#00C9A7","#F72585","#4CC9F0","#B5E48C",
]
PALETTE = DIGIT_COLORS

MODELS = {
    "Logistic Regression":  (LogisticRegression,  {"max_iter": 500, "random_state": 42, "solver": "saga"}),
    "SGD Classifier":       (SGDClassifier,        {"max_iter": 100, "random_state": 42, "tol": 1e-3}),
    "Decision Tree":        (DecisionTreeClassifier,{"random_state": 42}),
    "Random Forest":        (RandomForestClassifier,{"n_estimators": 100, "random_state": 42, "n_jobs": -1}),
    "K-Nearest Neighbors":  (KNeighborsClassifier,  {"n_neighbors": 5, "n_jobs": -1}),
    "SVM (RBF)":            (SVC,                   {"probability": True, "random_state": 42, "kernel": "rbf"}),
    "Naive Bayes":          (GaussianNB,             {}),
}

# ─────────────────────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────────────────────
def dark_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2d3154")
    return fig, ax

def style_ax(ax):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2d3154")

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="📦 Loading MNIST dataset...")
def load_mnist(n_samples):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y  = mnist.data, mnist.target.astype(int)
    # Stratified subsample
    idx = []
    per_class = n_samples // 10
    for digit in range(10):
        where = np.where(y == digit)[0]
        idx.extend(np.random.RandomState(42).choice(where, min(per_class, len(where)), replace=False))
    idx = np.array(idx)
    return X[idx].astype(np.float32) / 255.0, y[idx]

# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="🤖 Training models...")
def train_models(selected_models, n_samples, test_size, use_pca, pca_components, use_scaling):
    X, y = load_mnist(n_samples)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    if use_scaling:
        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

    if use_pca:
        pca_model = PCA(n_components=pca_components, random_state=42)
        X_tr = pca_model.fit_transform(X_tr)
        X_te = pca_model.transform(X_te)
    else:
        pca_model = None

    results = {}
    for name in selected_models:
        clf_cls, params = MODELS[name]
        clf = clf_cls(**params)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        results[name] = {
            "model":     clf,
            "y_pred":    y_pred,
            "accuracy":  accuracy_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred, average="weighted", zero_division=0),
            "recall":    recall_score(y_te, y_pred,    average="weighted", zero_division=0),
            "f1":        f1_score(y_te, y_pred,        average="weighted", zero_division=0),
            "cm":        confusion_matrix(y_te, y_pred),
        }
    return results, X_tr, X_te, y_tr, y_te, pca_model, X, y

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔢 MNIST Classifier Studio")
    st.markdown("---")

    st.markdown("**⚙️ Models**")
    selected_models = st.multiselect(
        "Choose models",
        list(MODELS.keys()),
        default=["Logistic Regression", "Random Forest", "K-Nearest Neighbors"],
    )

    st.markdown("---")
    st.markdown("**📦 Dataset**")
    n_samples = st.select_slider(
        "Training samples (total)",
        options=[2000, 5000, 10000, 20000, 40000, 70000],
        value=10000,
    )
    st.caption("⚠️ More samples = better accuracy but slower training.")
    test_size = st.slider("Test size (%)", 10, 30, 20, 5) / 100

    st.markdown("---")
    st.markdown("**🔬 Preprocessing**")
    use_scaling  = st.checkbox("StandardScaler", value=True)
    use_pca      = st.checkbox("PCA dimensionality reduction", value=True)
    pca_components = 50
    if use_pca:
        pca_components = st.slider("PCA components", 10, 150, 50, 10)
        st.caption("Reduces 784 → N dims. Speeds up training significantly.")

    st.markdown("---")
    st.markdown("**📊 Visualization**")
    show_cm       = st.checkbox("Confusion matrices",   value=True)
    show_roc      = st.checkbox("ROC curves",           value=True)
    show_proj     = st.checkbox("2D projection (PCA)",  value=True)
    show_tsne     = st.checkbox("t-SNE projection",     value=False)
    st.caption("⚠️ t-SNE is slow (~2 min for 2000 pts).")
    show_samples  = st.checkbox("Sample images",        value=True)
    show_errors   = st.checkbox("Misclassified images", value=True)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.title("🔢 MNIST Classifier Studio")
st.caption("Multi-model digit classification · confusion matrices · ROC · PCA/t-SNE projections · error analysis")
st.markdown("---")

if not selected_models:
    st.warning("👈 Please select at least one model from the sidebar.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────
with st.spinner("Loading & training — this may take a moment..."):
    results, X_tr_f, X_te_f, y_tr, y_te, pca_model, X_raw, y_raw = train_models(
        tuple(selected_models), n_samples, test_size,
        use_pca, pca_components, use_scaling,
    )

st.success(f"✅ {len(selected_models)} model(s) trained on {n_samples:,} samples "
           f"({'PCA ' + str(pca_components) + 'D' if use_pca else '784D raw'})")

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Performance",
    "🗺️ Decision Space",
    "📉 ROC Curves",
    "🔢 Confusion Matrices",
    "🖼️ Sample & Error Analysis",
    "🔍 Data Explorer",
])

# ════════════════════════════════════════
# TAB 1 — PERFORMANCE
# ════════════════════════════════════════
with tab1:
    # Metric cards
    cols = st.columns(len(selected_models))
    for i, name in enumerate(selected_models):
        r = results[name]
        with cols[i]:
            st.metric(f"🎯 {name}", f"{r['accuracy']:.2%}", f"F1: {r['f1']:.4f}")
            st.caption(
                f"Precision: {r['precision']:.4f}  |  "
                f"Recall: {r['recall']:.4f}"
            )

    st.markdown("---")

    # ── Grouped bar chart ──
    st.subheader("Comparative Metrics")
    metrics       = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    n  = len(selected_models)
    x  = np.arange(len(metric_labels))
    w  = 0.8 / n

    fig, ax = dark_fig(11, 5)
    for i, name in enumerate(selected_models):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * w - (n - 1) * w / 2, vals, w * 0.9,
                      label=name, color=PALETTE[i % len(PALETTE)], alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, color=MUTED, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", color=MUTED)
    ax.legend(facecolor=BG, edgecolor="#2d3154", labelcolor=MUTED, fontsize=8)
    ax.grid(axis="y", color="#1e2135", linewidth=0.7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Per-class F1 heatmap ──
    st.subheader("Per-Digit F1 Score")
    from sklearn.metrics import f1_score as f1_per

    per_class_f1 = {}
    for name in selected_models:
        per_class_f1[name] = f1_score(
            y_te, results[name]["y_pred"], average=None, labels=list(range(10)), zero_division=0
        )

    matrix = np.array([per_class_f1[n] for n in selected_models])

    fig2, ax2 = plt.subplots(figsize=(12, max(2, len(selected_models) * 0.8 + 1.5)))
    fig2.patch.set_facecolor(BG)
    style_ax(ax2)

    custom_cmap = LinearSegmentedColormap.from_list("c", [BG, "#1a3a4a", "#4fc3f7"])
    im = ax2.imshow(matrix, cmap=custom_cmap, aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(10))
    ax2.set_xticklabels([f"Digit {d}" for d in range(10)], color=MUTED, fontsize=8)
    ax2.set_yticks(range(len(selected_models)))
    ax2.set_yticklabels(selected_models, color=MUTED, fontsize=8)
    for r in range(len(selected_models)):
        for c in range(10):
            val = matrix[r, c]
            ax2.text(c, r, f"{val:.2f}", ha="center", va="center",
                     fontsize=8, color="white" if val < 0.6 else BG, fontweight="bold")
    cb = plt.colorbar(im, ax=ax2)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    ax2.set_title("F1 Score per digit per model", color=LIGHT, fontsize=10, pad=10)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ════════════════════════════════════════
# TAB 2 — DECISION SPACE (PCA / t-SNE)
# ════════════════════════════════════════
with tab2:
    st.markdown("""
    > MNIST has 784 dimensions — decision boundaries cannot be visualised directly.
    > We project the feature space to 2D using **PCA** and optionally **t-SNE**.
    """)

    # ── PCA 2D scatter ──
    if show_proj:
        st.subheader("PCA 2D Projection — Feature Space")

        @st.cache_data(show_spinner="Computing PCA 2D projection...")
        def get_pca2d(n_samples):
            X, y = load_mnist(n_samples)
            X_s  = StandardScaler().fit_transform(X)
            proj = PCA(n_components=2, random_state=42).fit_transform(X_s)
            return proj, y

        proj2d, y_proj = get_pca2d(min(n_samples, 5000))

        fig_p, ax_p = dark_fig(10, 6)
        for digit in range(10):
            mask = y_proj == digit
            ax_p.scatter(proj2d[mask, 0], proj2d[mask, 1],
                         c=DIGIT_COLORS[digit], s=10, alpha=0.55,
                         edgecolors="none", label=str(digit))
        ax_p.set_xlabel("PC1", color=MUTED, fontsize=9)
        ax_p.set_ylabel("PC2", color=MUTED, fontsize=9)
        ax_p.set_title(f"PCA 2D — {min(n_samples,5000):,} samples", color=LIGHT, fontsize=11)
        ax_p.legend(title="Digit", facecolor=BG, edgecolor="#2d3154",
                    labelcolor=MUTED, fontsize=7, title_fontsize=8,
                    ncol=2, markerscale=2)
        ax_p.grid(color="#1e2135", linewidth=0.4)
        plt.tight_layout()
        st.pyplot(fig_p)
        plt.close(fig_p)

    # ── Decision boundary on PCA-2D ──
    st.subheader("Decision Boundary on PCA 2D (Fast Models Only)")
    fast_models = [n for n in selected_models if n in (
        "Logistic Regression", "SGD Classifier", "Decision Tree", "Naive Bayes"
    )]

    if not fast_models:
        st.info("Select Logistic Regression, SGD, Decision Tree, or Naive Bayes to see decision boundaries.")
    else:
        @st.cache_data(show_spinner="Computing PCA-2D decision boundaries...")
        def get_boundary_data(n_samples, test_size, use_scaling, fast_models_tuple):
            X, y = load_mnist(n_samples)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            if use_scaling:
                sc   = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_te = sc.transform(X_te)
            pca2 = PCA(n_components=2, random_state=42)
            X_tr_2d = pca2.fit_transform(X_tr)
            X_te_2d = pca2.transform(X_te)
            return X_tr_2d, X_te_2d, y_tr, y_te

        X_tr_2d, X_te_2d, y_tr_2d, y_te_2d = get_boundary_data(
            n_samples, test_size, use_scaling, tuple(fast_models)
        )

        n_fm   = len(fast_models)
        n_cols = min(2, n_fm)
        n_rows = (n_fm + 1) // 2
        fig_b, axes_b = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
        fig_b.patch.set_facecolor(BG)

        if n_fm == 1:
            axes_b = [axes_b]
        elif n_rows == 1:
            axes_b = list(axes_b)
        else:
            axes_b = [ax for row in axes_b for ax in row]
        for j in range(n_fm, len(axes_b)):
            axes_b[j].set_visible(False)

        cmap_boundary = LinearSegmentedColormap.from_list(
            "digits", DIGIT_COLORS, N=10
        )

        for idx, name in enumerate(fast_models):
            ax  = axes_b[idx]
            style_ax(ax)

            clf_cls, params = MODELS[name]
            clf2d = clf_cls(**params)
            clf2d.fit(X_tr_2d, y_tr_2d)

            x_min, x_max = X_tr_2d[:, 0].min() - 1, X_tr_2d[:, 0].max() + 1
            y_min, y_max = X_tr_2d[:, 1].min() - 1, X_tr_2d[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                                 np.linspace(y_min, y_max, 250))
            Z = clf2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            ax.contourf(xx, yy, Z, alpha=0.25, cmap=cmap_boundary, levels=np.arange(-0.5, 10.5, 1))
            ax.contour(xx, yy, Z, colors=["#ffffff10"], linewidths=0.6, levels=np.arange(-0.5, 10.5, 1))

            # Scatter test points (subsample for speed)
            sub = min(1500, len(y_te_2d))
            rng = np.random.RandomState(0)
            si  = rng.choice(len(y_te_2d), sub, replace=False)
            for digit in range(10):
                m = y_te_2d[si] == digit
                ax.scatter(X_te_2d[si][m, 0], X_te_2d[si][m, 1],
                           c=DIGIT_COLORS[digit], s=8, alpha=0.7, edgecolors="none")

            ax.set_title(name, color=LIGHT, fontsize=10, pad=8)
            ax.set_xlabel("PC1", color=MUTED, fontsize=8)
            ax.set_ylabel("PC2", color=MUTED, fontsize=8)
            acc2d = accuracy_score(y_te_2d, clf2d.predict(X_te_2d))
            ax.set_title(f"{name}\n(2D acc: {acc2d:.2%})", color=LIGHT, fontsize=9, pad=8)

        plt.tight_layout()
        st.pyplot(fig_b)
        plt.close(fig_b)

    # ── t-SNE ──
    if show_tsne:
        st.subheader("t-SNE 2D Projection")
        tsne_n = st.slider("t-SNE sample size", 500, 3000, 1500, 500)

        @st.cache_data(show_spinner="Computing t-SNE (this takes ~1-2 min)...")
        def get_tsne(n):
            X, y = load_mnist(n)
            X_s  = StandardScaler().fit_transform(X)
            Xp   = PCA(n_components=50, random_state=42).fit_transform(X_s)
            proj = TSNE(n_components=2, random_state=42, perplexity=30,
                        n_iter=1000, init="pca").fit_transform(Xp)
            return proj, y

        tsne_proj, tsne_y = get_tsne(tsne_n)
        fig_t, ax_t = dark_fig(10, 7)
        for digit in range(10):
            mask = tsne_y == digit
            ax_t.scatter(tsne_proj[mask, 0], tsne_proj[mask, 1],
                         c=DIGIT_COLORS[digit], s=12, alpha=0.7,
                         edgecolors="none", label=str(digit))
        ax_t.set_title(f"t-SNE — {tsne_n:,} samples", color=LIGHT, fontsize=11)
        ax_t.set_xlabel("t-SNE 1", color=MUTED)
        ax_t.set_ylabel("t-SNE 2", color=MUTED)
        ax_t.legend(title="Digit", facecolor=BG, edgecolor="#2d3154",
                    labelcolor=MUTED, fontsize=7, ncol=2, markerscale=2)
        ax_t.grid(color="#1e2135", linewidth=0.4)
        plt.tight_layout()
        st.pyplot(fig_t)
        plt.close(fig_t)

# ════════════════════════════════════════
# TAB 3 — ROC CURVES
# ════════════════════════════════════════
with tab3:
    if show_roc:
        st.subheader("ROC Curves — One-vs-Rest per Digit")
        y_bin = label_binarize(y_te, classes=list(range(10)))

        selected_digit = st.selectbox("Select digit class", list(range(10)), index=0)

        fig_r, ax_r = dark_fig(10, 5)
        ax_r.plot([0, 1], [0, 1], "--", color="#3a3f5c", linewidth=1)

        for i, name in enumerate(selected_models):
            mdl = results[name]["model"]
            if hasattr(mdl, "predict_proba"):
                y_score = mdl.predict_proba(X_te_f)[:, selected_digit]
            elif hasattr(mdl, "decision_function"):
                df = mdl.decision_function(X_te_f)
                y_score = df[:, selected_digit] if df.ndim > 1 else df
            else:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, selected_digit], y_score)
            roc_auc     = auc(fpr, tpr)
            ax_r.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)],
                      linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")

        ax_r.set_title(f"ROC — Digit {selected_digit} vs Rest", color=LIGHT, fontsize=11)
        ax_r.set_xlabel("False Positive Rate", color=MUTED, fontsize=9)
        ax_r.set_ylabel("True Positive Rate",  color=MUTED, fontsize=9)
        ax_r.legend(facecolor=BG, edgecolor="#2d3154", labelcolor=MUTED, fontsize=8)
        ax_r.grid(color="#1e2135", linewidth=0.6)
        plt.tight_layout()
        st.pyplot(fig_r)
        plt.close(fig_r)

        # ── Mean AUC bar ──
        st.subheader("Mean AUC (macro-average across all digits)")
        fig_auc, ax_auc = dark_fig(8, 4)
        for i, name in enumerate(selected_models):
            mdl = results[name]["model"]
            aucs = []
            if hasattr(mdl, "predict_proba"):
                proba = mdl.predict_proba(X_te_f)
                for d in range(10):
                    if y_bin[:, d].sum() > 0:
                        fpr, tpr, _ = roc_curve(y_bin[:, d], proba[:, d])
                        aucs.append(auc(fpr, tpr))
            mean_auc = np.mean(aucs) if aucs else 0
            ax_auc.bar(name, mean_auc, color=PALETTE[i % len(PALETTE)], alpha=0.88)
            ax_auc.text(i, mean_auc + 0.003, f"{mean_auc:.4f}",
                        ha="center", fontsize=8, color=MUTED)

        ax_auc.set_ylim(0, 1.08)
        ax_auc.set_ylabel("Mean AUC", color=MUTED)
        ax_auc.tick_params(axis="x", colors=MUTED, rotation=20)
        ax_auc.grid(axis="y", color="#1e2135", linewidth=0.6)
        plt.tight_layout()
        st.pyplot(fig_auc)
        plt.close(fig_auc)
    else:
        st.info("Enable ROC Curves in the sidebar ↩")

# ════════════════════════════════════════
# TAB 4 — CONFUSION MATRICES
# ════════════════════════════════════════
with tab4:
    if show_cm:
        for name in selected_models:
            st.subheader(f"Confusion Matrix — {name}")
            cm = results[name]["cm"]

            fig_c, ax_c = plt.subplots(figsize=(9, 7))
            fig_c.patch.set_facecolor(BG)
            style_ax(ax_c)

            custom_cmap = LinearSegmentedColormap.from_list("cm", [BG, "#1a3a4a", "#4fc3f7"])
            im = ax_c.imshow(cm, cmap=custom_cmap, aspect="auto")
            ax_c.set_xticks(range(10))
            ax_c.set_yticks(range(10))
            ax_c.set_xticklabels([str(d) for d in range(10)], color=MUTED, fontsize=9)
            ax_c.set_yticklabels([str(d) for d in range(10)], color=MUTED, fontsize=9)
            ax_c.set_xlabel("Predicted", color=MUTED, fontsize=9)
            ax_c.set_ylabel("Actual",    color=MUTED, fontsize=9)
            ax_c.set_title(f"{name} — Accuracy {results[name]['accuracy']:.2%}",
                           color=LIGHT, fontsize=11, pad=10)

            thresh = cm.max() * 0.55
            for r in range(10):
                for c in range(10):
                    ax_c.text(c, r, str(cm[r, c]),
                              ha="center", va="center", fontsize=8,
                              color="white" if cm[r, c] < thresh else BG,
                              fontweight="bold")

            cb = plt.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
            cb.ax.tick_params(colors=MUTED, labelsize=7)
            plt.tight_layout()
            st.pyplot(fig_c)
            plt.close(fig_c)

            with st.expander(f"📋 Classification Report — {name}"):
                st.code(
                    classification_report(y_te, results[name]["y_pred"],
                                          target_names=[str(d) for d in range(10)]),
                    language=None,
                )
    else:
        st.info("Enable Confusion Matrices in the sidebar ↩")

# ════════════════════════════════════════
# TAB 5 — SAMPLE & ERROR ANALYSIS
# ════════════════════════════════════════
with tab5:
    # ── Sample digit images ──
    if show_samples:
        st.subheader("Sample Images from Dataset")
        X_raw_orig, y_raw_orig = load_mnist(n_samples)

        fig_s, axes_s = plt.subplots(2, 10, figsize=(14, 3.5))
        fig_s.patch.set_facecolor(BG)
        for digit in range(10):
            indices = np.where(y_raw_orig == digit)[0]
            for row, idx in enumerate(indices[:2]):
                ax = axes_s[row, digit]
                ax.imshow(X_raw_orig[idx].reshape(28, 28), cmap="inferno", interpolation="nearest")
                ax.axis("off")
                if row == 0:
                    ax.set_title(str(digit), color=DIGIT_COLORS[digit], fontsize=10, pad=4)
        fig_s.suptitle("Two samples per digit class", color=LIGHT, fontsize=11, y=1.02)
        plt.tight_layout()
        st.pyplot(fig_s)
        plt.close(fig_s)

    # ── Misclassified images ──
    if show_errors:
        st.subheader("Misclassified Images")
        error_model = st.selectbox("Select model", selected_models, key="error_model")

        # Recover original test images
        @st.cache_data
        def get_test_images(n_samples, test_size):
            X, y = load_mnist(n_samples)
            _, X_te_raw, _, y_te_raw = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            return X_te_raw, y_te_raw

        X_te_raw, y_te_raw = get_test_images(n_samples, test_size)
        y_pred_err = results[error_model]["y_pred"]
        wrong_idx  = np.where(y_te_raw != y_pred_err)[0]

        n_show = min(40, len(wrong_idx))
        st.caption(f"Showing {n_show} of {len(wrong_idx)} misclassified images "
                   f"({len(wrong_idx)/len(y_te_raw):.1%} error rate)")

        cols_per_row = 10
        n_rows_err   = (n_show + cols_per_row - 1) // cols_per_row
        fig_e, axes_e = plt.subplots(n_rows_err, cols_per_row,
                                     figsize=(14, 1.8 * n_rows_err))
        fig_e.patch.set_facecolor(BG)

        for j in range(n_rows_err * cols_per_row):
            r_, c_ = divmod(j, cols_per_row)
            ax = axes_e[r_, c_] if n_rows_err > 1 else axes_e[c_]
            if j < n_show:
                idx = wrong_idx[j]
                ax.imshow(X_te_raw[idx].reshape(28, 28), cmap="inferno", interpolation="nearest")
                ax.set_title(f"T:{y_te_raw[idx]}\nP:{y_pred_err[idx]}",
                             fontsize=6, color="#FF6B6B", pad=2)
            ax.axis("off")

        fig_e.suptitle(f"Misclassified — {error_model}", color=LIGHT, fontsize=10, y=1.01)
        plt.tight_layout()
        st.pyplot(fig_e)
        plt.close(fig_e)

        # ── Error heatmap per digit ──
        st.subheader("Error Rate per Digit")
        fig_er, ax_er = dark_fig(10, 4)
        for digit in range(10):
            mask    = y_te_raw == digit
            n_wrong = np.sum(y_pred_err[mask] != digit)
            n_total = mask.sum()
            err_rate = n_wrong / n_total if n_total > 0 else 0
            ax_er.bar(digit, err_rate, color=DIGIT_COLORS[digit], alpha=0.85)
            ax_er.text(digit, err_rate + 0.002, f"{err_rate:.1%}",
                       ha="center", fontsize=8, color=MUTED)

        ax_er.set_xticks(range(10))
        ax_er.set_xticklabels([f"Digit {d}" for d in range(10)], color=MUTED, rotation=30, fontsize=8)
        ax_er.set_ylabel("Error Rate", color=MUTED)
        ax_er.set_ylim(0, min(1.0, ax_er.get_ylim()[1] + 0.05))
        ax_er.grid(axis="y", color="#1e2135", linewidth=0.6)
        ax_er.set_title(f"Error rate per digit — {error_model}", color=LIGHT, fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_er)
        plt.close(fig_er)

# ════════════════════════════════════════
# TAB 6 — DATA EXPLORER
# ════════════════════════════════════════
with tab6:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples Used", f"{n_samples:,}")
    c2.metric("Training Samples",   f"{int(n_samples * (1 - test_size)):,}")
    c3.metric("Test Samples",       f"{int(n_samples * test_size):,}")
    c4.metric("Input Dimensions",   f"{pca_components if use_pca else 784}")

    st.markdown("---")

    # ── Pixel intensity distribution ──
    st.subheader("Pixel Intensity Distribution per Digit")
    fig_px, axes_px = plt.subplots(2, 5, figsize=(14, 5))
    fig_px.patch.set_facecolor(BG)

    X_raw_d, y_raw_d = load_mnist(n_samples)
    for digit in range(10):
        r_, c_ = divmod(digit, 5)
        ax = axes_px[r_, c_]
        style_ax(ax)
        mask = y_raw_d == digit
        vals = X_raw_d[mask].ravel()
        ax.hist(vals, bins=30, color=DIGIT_COLORS[digit], alpha=0.75, density=True)
        ax.set_title(f"Digit {digit}", color=DIGIT_COLORS[digit], fontsize=9)
        ax.set_xlabel("Pixel value", color=MUTED, fontsize=7)
        ax.grid(color="#1e2135", linewidth=0.4)

    fig_px.suptitle("Normalised pixel intensity distribution per class",
                    color=LIGHT, fontsize=10, y=1.01)
    plt.tight_layout()
    st.pyplot(fig_px)
    plt.close(fig_px)

    # ── Average digit images ──
    st.subheader("Average Image per Digit")
    fig_avg, axes_avg = plt.subplots(1, 10, figsize=(14, 2))
    fig_avg.patch.set_facecolor(BG)

    for digit in range(10):
        mask   = y_raw_d == digit
        avg_img = X_raw_d[mask].mean(axis=0).reshape(28, 28)
        ax = axes_avg[digit]
        ax.imshow(avg_img, cmap="inferno", interpolation="nearest")
        ax.axis("off")
        ax.set_title(str(digit), color=DIGIT_COLORS[digit], fontsize=10)

    fig_avg.suptitle("Mean pixel image per digit class", color=LIGHT, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig_avg)
    plt.close(fig_avg)

    # ── Class balance ──
    st.subheader("Class Distribution")
    fig_cb, ax_cb = dark_fig(9, 4)
    counts = [np.sum(y_raw_d == d) for d in range(10)]
    ax_cb.bar(range(10), counts, color=DIGIT_COLORS, alpha=0.88)
    for i, cnt in enumerate(counts):
        ax_cb.text(i, cnt + 5, str(cnt), ha="center", fontsize=8, color=MUTED)
    ax_cb.set_xticks(range(10))
    ax_cb.set_xticklabels([f"Digit {d}" for d in range(10)], color=MUTED, rotation=30, fontsize=8)
    ax_cb.set_ylabel("Count", color=MUTED)
    ax_cb.grid(axis="y", color="#1e2135", linewidth=0.6)
    plt.tight_layout()
    st.pyplot(fig_cb)
    plt.close(fig_cb)
