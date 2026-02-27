# Namaskaar !!

Welcome to my learning journey. 

**My Mission:** Move beyond the "surface level" wrapper-apps and mystery-box LLMs. My goal is to master the mathematical foundations, the architectural "black magic" of Deep Learning, and eventually build a complete Large Language Model (LLM) from scratch. 

Basically, I’m learning how the "magic" works so I don't have to rely on a wand.

---

## Tech Stack & Environment
This repository uses a **Multi-package Workspace** managed by [uv](https://docs.astral.sh), the extremely fast Python package manager.

- **Language:** Python 3.13+
- **Environment:** uv Workspaces
- **IDEs:** PyCharm / VS Code / Jupyter

---

## Project Roadmap

### Course Labs (`projects/course_labs`)
*Core mathematical implementations learnt from the Stanford University Specialization course.*

- [x] **Linear Regression (Single Variable):** Cost functions & Gradient Descent.
- [x] **Multiple Linear Regression:** Vectorized Gradient Descent & Feature Scaling.
- [ ] **Logistic Regression:** Classification & Decision Boundaries.
- [ ] **Regularization:** Overfitting prevention ($L1$/$L2$).
- [ ] **Neural Networks:** Backpropagation and Activation Functions.

### Kaggle Challenges 

*Applying course theory to real-world datasets.*

| Status | Phase | Dataset | Type | Concepts | Misc Learning | Project Link |
| :---: | :---: | :--- | :---: |:---------------------------|:-------------------------------------| :--- |
| [x] | **I** | Student Performance | Reg | Gradient Descent Basics    | Simple linear correlation            | [View Code](./projects/linear-regression/student-performance) |
| [x] | **I** | Medical Insurance | Reg | Categorical Variables      | Converting "Smoker" to 0/1           | [View Code](./projects/linear-regression/medical-insurance-prediction) |
| [ ] | **I** | Ames Housing Data | Reg | Feature Engineering        | Handling skewness & missing data     | [View Code](./projects/linear-regression/ames-housing) |
| [ ] | **II** | [Iris Flower](https://www.kaggle.com) | Class | **Binary vs. Multi-class** | Implementing One-vs-Rest logic       | [View Code](./projects/classification/iris-flower) |
| [ ] | **II** | [Titanic Disaster](https://www.kaggle.com) | Class | **Data Imputation**        | Filling missing Age/Cabin values     | [View Code](./projects/classification/titanic-survival) |
| [ ] | **II** | [Pima Diabetes](https://www.kaggle.com) | Class | **Numerical Stability**    | $z$-score Scaling (Critical for GD)  | [View Code](./projects/classification/pima-diabetes) |
| [ ] | **III** | [Breast Cancer](https://www.kaggle.com) | Class | **Multicollinearity**      | Removing redundant features          | [View Code](./projects/classification/breast-cancer) |
| [ ] | **III** | [Bank Marketing](https://www.kaggle.com) | Class | **Evaluation Metrics**     | Precision, Recall, & F1-Score        | [View Code](./projects/classification/bank-marketing) |
| [ ] | **III** | [Red Wine Quality](https://www.kaggle.com) | Class | **Regularization**         | Implementing $L_2$ (Ridge) penalties | [View Code](./projects/classification/wine-quality) |
| [ ] | **IV** | [SMS Spam](https://www.kaggle.com) | Class | **NLP Vectorization**      | Converting text to math (TF-IDF)     | [View Code](./projects/classification/sms-spam) |
| [ ] | **IV** | [Credit Card Fraud](https://www.kaggle.com) | Class | **Extreme Imbalance**      | AUC-ROC and 0.17% detection          | [View Code](./projects/classification/credit-fraud) |

---

## Tracking My Progress
- [ ] **Phase 1: Statistical Foundations** (Linear/Logistic Regression, SVMs) - *CURRENT*
- [ ] **Phase 2: Deep Learning** (Neural Networks, CNNs, RNNs)
- [ ] **Phase 3: Sequence Models** (LSTMs, GRUs, Attention Mechanisms)
- [ ] **Phase 4: Transformers** (Self-attention, Multi-head attention, Positional Encoding)
- [ ] **Phase 5: The LLM** (Pre-training, Fine-tuning, RLHF)

---

> "If you can't build it from scratch, you don't actually understand it." — Someone smart, probably.
