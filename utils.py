import shap

def explain_shap(model, image, text):
    # Generate SHAP values (for text and image)
    explainer = shap.Explainer(model)
    shap_values = explainer([image, text])
    return shap_values
