
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model, test_ds, model_name="Model"):
    """
    Évalue un modèle sur le dataset de test.
    Les classes doivent être encodées : NORMAL=0, PNEUMONIA=1.
    """

    y_true = np.concatenate([
        labels.numpy().reshape(-1)
        for _, labels in test_ds
    ]).astype(int)

    y_pred_proba = model.predict(test_ds, verbose=1).reshape(-1)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    print(f"\n{'=' * 50}")
    print(f"ÉVALUATION : {model_name}")
    print(f"{'=' * 50}")

    print("\nRapport de classification :")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["NORMAL", "PNEUMONIA"],
            zero_division=0
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("Matrice de confusion :")
    print(cm)

    auc = roc_auc_score(y_true, y_pred_proba)
    print(f"\nROC-AUC : {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NORMAL", "PNEUMONIA"]
    ).plot(ax=axes[0], cmap="Blues", colorbar=False)

    axes[0].set_title(f"Matrice de confusion - {model_name}")

    axes[1].plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"Courbe ROC - {model_name}")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": float(
            __import__("sklearn.metrics", fromlist=["precision_score"])
            .precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            __import__("sklearn.metrics", fromlist=["recall_score"])
            .recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            __import__("sklearn.metrics", fromlist=["f1_score"])
            .f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist()
    }
