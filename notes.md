My lecture notes documented from the ./2026_AIBIS_Exercise2_XAI_Kickoff-Session_V1.pdf presentation:

Presentation + Code/Notebook
Passau MedCo: offers medical devices & offers consulting services
Experiencing increasing employee attrition

Fictional Setting: no one size fits all solution
Explainable solution: needs to provide explanation on how prediction was decided

LIME & SHAP explainable AI methods
LIME: is locally but there are techniques to get global

**Get to the point**, explain what is necessary and have additional details for technical people.

1.1 any patterns, can we group clusters, any outliers. Exploratory analysis
1.2 train model to predict if employee leaves
1.3 Apply SHAP or LIME + a global explanation method

Don’t overengineer data preparation & analysis
Won’t challenge on Classification Algorithm & Hyperparameters. Just specify and explain briefly (why did you pick the algorithm needs to fit the data, did you do some optimization?)

Provide 3 local explanations: 3 cases that are different from each other, 

- “employee was part of attribution group”
- “one employee who stayed for clear reasons”
- “one employee who is a border case”

Which features are used in the prediction: local & global level

Prefer SHAP, natively supports global & local explanations

1. look at dataset, which algorithms are suitable? How much effort to spend?

Showcase PoC: show machine learning approach, pull up Jupiter notebook: Program dashboard, use Plotly Dash or Streamlit
OpenXAI provides LIME & SHAP