# SecondDraftML

Accelerate your machine learning workflow by doing 50% of your modeling iterations in 10% of the time. SecondDraftML is a visual insight-oriented AutoML package designed for rapid exploration and model prototyping, helping you quickly navigate the critical first half of the ML workflow.

## Why SecondDraftML?

The typical ML development workflow involves:
   - Prototyping
   - Deployment (often a separate time-consuming process that can be skipped until collecting business team feedback)
   - Monitoring and educating business teams on proper usage

In real-world business scenarios, machine learning success is rarely about marginal performance gains. Instead, what matters most is:

1. **Integration & Usability Over Marginal Gains** - The 0.00001% performance improvement that consumes significant time often means little to business outcomes. What truly matters is seamless integration with broader systems and ease of use by business teams (learning curve).

2. **Multiple Formulations, Rapid Iteration** - Business problems can often be formulated multiple ways (binary classification predicting improvement vs. regression predicting exact values, etc.). You rarely know which approach best suits business needs upfront. SecondDraftML enables you to quickly present multiple approaches, collect feedback, and iterate - making rapid prototyping a game changer for business alignment.

3. **Beyond the Hype: Tabular Data Dominance** - Despite the generative AI/LLM hype, most real-world ML problems remain tabular dataset regression or time series problems. SecondDraftML focuses on these practical, everyday business challenges that constitute the majority of production ML systems.

## Features

- **Automated Exploratory Data Analysis (EDA)**
  - Visual null analysis with interactive charts
  - Correlation matrices and pairwise relationship visualizations
  - Feature distribution plots (histograms, box plots, KDE)
  - Automatic data profiling and statistical summaries

- **Rapid Model Prototyping**
  - Train multiple model types with default parameters:
    - Linear Models (Linear Regression, Logistic Regression)
    - Tree-Based Models (Random Forest, XGBoost, LightGBM, CatBoost)
    - Support Vector Machines
    - Neural Networks (MLP)
    - And more...
  - Automated hyperparameter tuning with visual comparison

- **Visual Model Evaluation**
  - Model performance metrics visualization
  - Feature importance charts across multiple models
  - Model correlation analysis (how similar are your models' predictions?)
  - Interactive leaderboard with model comparison capabilities

- **Interactive Chainlit Interface**
  - Visual workflow that guides through the ML process
  - One-click model training and evaluation
  - Interactive visualizations with matplotlib
  - Export results and charts with ease

## Installation

```bash
pip install seconddraftml
```

## Quick Start

```python
from seconddraftml import MetaRegressor

meta_regressor = MetaRegressor(
    nmodels,  # Number of models to train
    nfolds, # Number of folds for cross-validation
    split_feature, # Feature to split data on (e.g., 'date' for time series)
    split_ratio, # Ratio to split data into training and validation sets
    verbose=True,  # Verbosity level
)

meta_regressor.fit(X_train, y_train)  # Fit the model on training data
predictions = meta_regressor.predict(X_test)  # Make predictions on test data
```

Alternatively, use the command line interface that runs the Chainlit UI:

```bash
seconddraftml ui
```

## Usage

1. **Data Loading**: Load your dataset through the UI or specify it in code
2. **Data Profiling**: Explore automatic EDA visualizations and data insights
3. **Model Training**: Select models to train or use the default comprehensive set
4. **Evaluation**: Compare model performance through interactive visualizations
5. **Iteration**: Use insights to refine your features and model selection

## Supported Visualization Types

- Nullity matrix and bar charts
- Correlation heatmaps and scatter matrices
- Feature distribution plots
- Model performance comparison charts
- Feature importance plots (permutation, SHAP, and native)
- Residual analysis plots
- Learning curves and validation curves
- Model prediction correlation matrices

## Contributing

We welcome contributions! Please see our contributing guidelines for details.
