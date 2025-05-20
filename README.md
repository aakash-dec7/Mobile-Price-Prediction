# Mobile Price Prediction

This repository aims to build a **Machine Learning** model that can predict the price range of mobile phones based on their technical specifications and features. With the growing variety of smartphones in the market, consumers and retailers can benefit from a system that estimates price categories - ranging from low-cost to very high-end—based solely on hardware and connectivity features.

## Model Architecture

This project leverages three powerful ensemble learning classifiers - `RandomForestClassifier`, `XGBClassifier`, and `AdaBoostClassifier` - to achieve robust performance on the given classification task. Each model has been configured with a well-curated hyperparameter grid for optimal tuning using techniques like `GridSearchCV`.

- ### RandomForestClassifier

    Random Forest is an ensemble of decision trees, where each tree is trained on a bootstrapped sample of the dataset and uses a random subset of features for splitting. It improves accuracy by reducing overfitting and variance.

    **Hyperparameters tuned:**

  - `n_estimators`: Number of trees in the forest.

  - `max_depth`: Maximum depth of each tree.

  - `min_samples_split`: Minimum samples required to split a node.

  - `min_samples_leaf`: Minimum samples required at a leaf node.

  - `max_features`: Number of features to consider when looking for the best split (`sqrt` or `log2`).

- ### XGBClassifier (XGBoost)

    XGBoost (Extreme Gradient Boosting) is a highly efficient and scalable implementation of gradient boosting. It builds models in a sequential manner where each new tree corrects errors made by the previous ones.

    **Hyperparameters tuned:**

  - `n_estimators`: Number of boosting rounds.

  - `max_depth`: Maximum tree depth for base learners.

  - `learning_rate`: Step size shrinkage to prevent overfitting.

  - `gamma`: Minimum loss reduction required to make a split.

  - `reg_lambda`: L2 regularization term to reduce model complexity.

- ### AdaBoostClassifier

    AdaBoost (Adaptive Boosting) combines multiple weak learners (typically decision stumps) in a way that focuses more on the misclassified instances by adjusting their weights iteratively.

    **Hyperparameters tuned:**

  - `n_estimators`: Number of weak learners to train.

  - `learning_rate`: Shrinks the contribution of each learner to control overfitting.

## Dataset

This dataset contains specifications of mobile phones, designed for classification into one of four price range categories (0 to 3), where 0 represents the lowest price range and 3 the highest. Each row represents a different mobile phone and includes a variety of hardware and software features that influence its market price.

**Key Features:**

- **Target Variable:**

  - `price_range` (0 to 3) — Indicates the price category of the mobile phone (0: Low Cost, 3: High Cost)

- **Performance & Hardware:**

  - `battery_power`: Battery capacity in mAh

  - `ram`: Random Access Memory in MB

  - `clock_speed`: Speed of the processor in GHz

  - `n_cores`: Number of processor cores

  - `int_memory`: Internal storage in GB

  - `mobile_wt`: Weight of the mobile phone in grams

- **Camera Features:**

  - `fc`: Front camera megapixels

  - `pc`: Primary camera megapixels

- **Display & Screen:**

  - `px_height`, `px_width`: Screen resolution in pixels

  - `sc_h`, `sc_w`: Screen height and width in cm

  - `m_dep`: Mobile depth (thickness) in cm

- **Connectivity & Features:**

  - `dual_sim`: Supports dual SIM (1: Yes, 0: No)

  - `four_g`, `three_g`: 4G and 3G connectivity support

  - `touch_screen`: Touch screen feature availability

  - `wifi`, `blue`: Wi-Fi and Bluetooth support

  - `talk_time`: Maximum talk time in hours

## Model Training Metrics

The following results summarize the best cross-validation scores and corresponding hyperparameters for each classifier:

```json
[
    {
        "model": "RandomForestClassifier",
        "best_params": {
            "classifier__max_depth": 10,
            "classifier__max_features": "sqrt",
            "classifier__min_samples_leaf": 1,
            "classifier__min_samples_split": 5,
            "classifier__n_estimators": 100
        },
        "best_score": 0.8785714285714284
    },
    {
        "model": "XGBClassifier",
        "best_params": {
            "classifier__gamma": 0,
            "classifier__learning_rate": 0.3,
            "classifier__max_depth": 3,
            "classifier__n_estimators": 50,
            "classifier__reg_lambda": 2
        },
        "best_score": 0.9071428571428571
    },
    {
        "model": "AdaBoostClassifier",
        "best_params": {
            "classifier__learning_rate": 0.1,
            "classifier__n_estimators": 50
        },
        "best_score": 0.7407142857142858
    }
]
```

## Model Evaluation Metrics

The following results summarize the model evaluation performance for the **best model** from `GridSearchCV`:

```json
{
    "accuracy": 0.735,
    "precision": 0.7275143624625188,
    "recall": 0.735,
    "f1_score": 0.7275006634506502,
}
```

## Installation

Clone the repository:

```sh
git clone https://github.com/aakash-dec7/Mobile-Price-Prediction.git
cd Mobile-Price-Prediction
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Initiate DVC

```sh
dvc init
```

### Run the pipeline

```sh
dvc repro
```

The pipeline automatically launches the Flask application at:

```text
http://localhost:3000/
```

## Conclusion

This project demonstrates the effectiveness of ensemble learning methods in predicting mobile phone price ranges based on hardware and connectivity features. Among the models evaluated, XGBoost achieved the highest performance with a cross-validation accuracy of over 90%, highlighting its strength in capturing complex patterns within the data.

By leveraging well-tuned models and a reproducible pipeline powered by DVC, this system provides a scalable foundation for price prediction tasks in e-commerce, retail analytics, and recommendation engines. Future improvements could include feature engineering, model ensembling, or incorporating more advanced deep learning techniques for further accuracy gains.

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
