import shap
import matplotlib.pyplot as plt
import os

def run_shap_analysis(pipeline, X_sample, max_display=15, save_dir="shap_outputs"):
    """
    Run SHAP explainability on the trained pipeline and save plots.

    Args:
        pipeline : trained sklearn Pipeline (with preprocessor + model)
        X_sample : DataFrame or array -> subset of features
        max_display : int -> number of features to show in SHAP summary plot
        save_dir : directory to save SHAP plots

    Returns:
        shap_values
    """

    os.makedirs(save_dir, exist_ok=True)

    classifier = pipeline.named_steps["model"]

    if "preprocessor" in pipeline.named_steps:
        X_transformed = pipeline.named_steps["preprocessor"].transform(X_sample)
    else:
        X_transformed = X_sample

    if hasattr(classifier, "predict_proba"):
        try:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer(X_transformed)
        except:
            explainer = shap.Explainer(classifier, X_transformed)
            shap_values = explainer(X_transformed)
    else:
        explainer = shap.LinearExplainer(classifier, X_transformed)
        shap_values = explainer(X_transformed)

    # --- Plot 1: Summary Plot (Feature Importance)
    plt.title("SHAP Feature Importance", fontsize=14)
    shap.summary_plot(shap_values, X_transformed, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_summary.png"), dpi=300)
    plt.show()

    # --- Plot 2: Bar Plot
    shap.summary_plot(shap_values, X_transformed, plot_type="bar", max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_bar.png"), dpi=300)
    plt.show()

    print(f" SHAP plots saved in: {save_dir}")

    return shap_values


