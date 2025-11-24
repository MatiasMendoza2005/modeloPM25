import shap
import torch
import numpy as np

def explain_model(model, sample):
    model.eval()

    def f(x):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            return model(x).detach().numpy()

    explainer = shap.KernelExplainer(f, sample[:50])
    shap_values = explainer.shap_values(sample[:10])

    shap.summary_plot(shap_values, sample[:10])
