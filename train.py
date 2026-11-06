"""Train the CNN and logistic regression baseline, calibrate, evaluate on the test set, build pools.

Usage: python train.py
"""
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from sudoku.calibration import ece, fit_temperature, nll, softmax
from sudoku.config import MODELS, RESULTS, TEAM_SEED, TRAIN_SEED
from sudoku.data import load_processed
from sudoku.eval import plot_confusion, plot_reliability
from sudoku.mosaic import build_pools, save_pools
from sudoku.nn import DigitReader, predict_logits, train_model


def scores(probs, y):
    return {"accuracy": float((probs.argmax(axis=1) == y).mean()), "ece": float(ece(probs, y)),
            "nll": nll(probs, y)}


def main():
    data = load_processed()
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    y_val, y_test = data["y_val"] - 1, data["y_test"] - 1

    model, history = train_model(data, seed=TRAIN_SEED)
    torch.save(model.state_dict(), MODELS / "cnn.pt")

    temperature = fit_temperature(predict_logits(model, data["x_val"]), y_val)
    (MODELS / "calibration.json").write_text(json.dumps({"temperature": temperature}))
    print(f"Temperature fitted on validation: {temperature:.3f}")

    logits = predict_logits(model, data["x_test"])
    raw, calibrated = softmax(logits), softmax(logits, temperature)

    print("Training logistic regression baseline...")
    baseline = LogisticRegression(max_iter=100, random_state=TRAIN_SEED)
    baseline.fit(data["x_train"].reshape(len(y_train := data["y_train"]), -1), y_train - 1)
    logreg = baseline.predict_proba(data["x_test"].reshape(len(y_test), -1))

    metrics = {"temperature": temperature, "history": history,
               "cnn_uncalibrated": scores(raw, y_test), "cnn_calibrated": scores(calibrated, y_test),
               "logistic_regression": scores(logreg, y_test)}

    metrics["confusion_matrix"] = {
        "cnn_calibrated": plot_confusion(calibrated, y_test, RESULTS / "confusion_matrix.png", "Calibrated CNN"),
        "logistic_regression": plot_confusion(logreg, y_test, RESULTS / "confusion_matrix_logreg.png",
                                              "Logistic regression")}
    plot_reliability({"CNN, uncalibrated": raw, "CNN, calibrated": calibrated,
                      "Logistic regression": logreg}, y_test, RESULTS / "reliability.png")

    pools = build_pools(data["x_test"], data["y_test"], DigitReader(), TEAM_SEED)
    save_pools(pools)
    metrics["pools"] = pools["summary"]
    (RESULTS / "nn_metrics.json").write_text(json.dumps(metrics, indent=2))
    for name in ("cnn_uncalibrated", "cnn_calibrated", "logistic_regression"):
        print(name, metrics[name])
    print("Pools:", pools["summary"])
    min_digit = float(np.min(data["x_test"].mean(axis=(1, 2))))
    print(f"Dimmest test digit mean brightness: {min_digit:.4f}")


if __name__ == "__main__":
    main()
