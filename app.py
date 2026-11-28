"""Management dashboard for the attrition PoC. Run: uv run python app.py -> http://127.0.0.1:8050

Uses the exported model (model.joblib) and the same data preparation and split as goal-1-2/goal-1-3.
All model numbers refer to the 294 test employees the model never saw during training.
"""
import re

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from dash import Dash, Input, Output, ctx, dcc, html
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- data & model (as in goal-1-3.ipynb)
SEED = 67
ONEHOT = ["Department", "EducationField", "JobRole", "MaritalStatus"]
TRAVEL = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
TRAVEL_LABEL = {0: "None", 1: "Rarely", 2: "Frequently"}
NAMES = {"OverTime": "Overtime", "YearsWithCurrManager": "Years with current manager",
         "NumCompaniesWorked": "Previous employers"}
RATING = {1: "Low", 2: "Medium", 3: "High", 4: "Very high"}
LEVELS = {"EnvironmentSatisfaction": RATING, "JobSatisfaction": RATING, "RelationshipSatisfaction": RATING,
          "JobInvolvement": RATING, "WorkLifeBalance": {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}}
BANDS = {"Age": [17, 25, 30, 35, 45, 60], "MonthlyIncome": [1000, 2500, 4000, 6000, 10000, 20000],
         "TotalWorkingYears": [-1, 2, 5, 10, 20, 40], "YearsAtCompany": [-1, 1, 3, 5, 10, 40],
         "YearsWithCurrManager": [-1, 0, 2, 5, 17], "DistanceFromHome": [0, 5, 10, 20, 29]}
LEVERS = ["OverTime", "BusinessTravel", "MonthlyIncome", "PercentSalaryHike", "JobLevel", "StockOptionLevel",
          "YearsSinceLastPromotion", "YearsWithCurrManager", "TrainingTimesLastYear", "EnvironmentSatisfaction",
          "JobSatisfaction", "WorkLifeBalance", "RelationshipSatisfaction", "JobInvolvement"]
CASES = {"Clear leaver": 688, "Clear stayer": 1179, "Border case": 1167}
GROUPINGS = ["Department", "JobRole", "OverTime", "BusinessTravel", "MaritalStatus", "Age", "MonthlyIncome",
             "YearsAtCompany"]

RAW = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")


def prepare(raw):
    """Encodes the raw data exactly like goal-1-2.ipynb."""
    df = raw.drop(columns=["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"])
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
    df["OverTime"] = (df["OverTime"] == "Yes").astype(int)
    df["Gender"] = (df["Gender"] == "Male").astype(int)
    df["BusinessTravel"] = df["BusinessTravel"].map(TRAVEL)
    return pd.get_dummies(df, columns=ONEHOT, dtype=int)


def source_of(column):
    """Returns the original feature a model column belongs to."""
    return next((f for f in ONEHOT if column.startswith(f + "_")), column)


def readable(feature):
    """Turns 'MonthlyIncome' into 'Monthly income'."""
    return NAMES.get(feature, re.sub(r"(?<!^)(?=[A-Z])", " ", feature).capitalize())


def shown(row, feature):
    """Returns the value of one feature of a model row as readable text."""
    if feature in ONEHOT:
        return next(c.split("_", 1)[1] for c in row.index if c.startswith(feature + "_") and row[c] == 1)
    value = row[feature]
    if feature in ("OverTime", "Gender"):
        return {"OverTime": ["No", "Yes"], "Gender": ["Female", "Male"]}[feature][int(value)]
    if feature == "BusinessTravel":
        return TRAVEL_LABEL[int(value)]
    return LEVELS[feature][int(value)] if feature in LEVELS else f"{value:,.0f}"


MODEL = joblib.load("model.joblib")
RF, TUNED = MODEL.estimator_, float(MODEL.best_threshold_)
DATA = prepare(RAW)
X, Y = DATA.drop(columns="Attrition"), DATA["Attrition"]
_, X_TEST, _, Y_TEST = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=SEED)
assert list(X_TEST.columns) == list(RF.feature_names_in_), "data does not match the exported model"
assert set(CASES.values()) <= set(X_TEST.index), "preset employees are not in the test set"

RISK = pd.Series(RF.predict_proba(X_TEST)[:, 1], index=X_TEST.index)
SOURCES = [source_of(c) for c in X_TEST.columns]
EXPLAINER = shap.TreeExplainer(RF)


def explain(rows):
    """Returns the base value and the SHAP values per original feature (one-hot columns summed)."""
    sv = EXPLAINER(rows)[:, :, 1]
    grouped = pd.DataFrame(sv.values, index=rows.index, columns=rows.columns).T.groupby(SOURCES, sort=False).sum().T
    return float(sv.base_values[0]), grouped


BASE, SHAP = explain(X_TEST)
assert np.abs(BASE + SHAP.sum(1) - RISK).max() < 1e-6, "SHAP values do not add up to the model output"
IMPORTANCE = SHAP.abs().mean().sort_values(ascending=False) * 100
BASE_RATE = Y.mean()

# ---------------------------------------------------------------- look
BLUE, ORANGE, GRAY = "#257de0", "#f47230", "#b8b8b8"
INK, MUTED, GRID = "#191919", "#666666", "#e7e7e7"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
PLOT = {"displayModeBar": False}


def style(fig, height=320, **layout):
    """Applies the shared chart look."""
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=30, t=10, b=10),
                      font=dict(family=FONT, size=13, color=INK), paper_bgcolor="white", plot_bgcolor="white",
                      hoverlabel=dict(bgcolor="white", font_color=INK), bargap=0.35,
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0), **layout)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, automargin=True, tickfont_color=MUTED, title_font_color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, automargin=True, tickfont_color=MUTED, title_font_color=MUTED,
                     ticksuffix="  ")
    return fig


def tile(value, label, note=""):
    return html.Div([html.Div(value, className="num"), html.Div(label, className="lab"),
                     html.Div(note, className="note")], className="tile")


def section(question, *children):
    return html.Section([html.H2(question), *children])


# ---------------------------------------------------------------- 2 workforce
def workforce_fig(feature):
    """Treemap of all 1,470 employees: cell size = headcount, green = below average attrition, red = above."""
    groups = pd.cut(RAW[feature], BANDS[feature]) if feature in BANDS else RAW[feature]
    if feature in BANDS:
        groups = groups.map(lambda b: f"{b.left + 1:,.0f}–{b.right:,.0f}")
    if feature == "OverTime":
        groups = groups.map({"Yes": "Overtime", "No": "No overtime"})
    cells = Y.groupby(groups.astype(str).str.replace("_", " ").values).agg(rate="mean", n="size", left="sum")
    cells = cells.reset_index(names="label").assign(id=lambda d: d["label"], parent="all")
    root = pd.DataFrame([{"id": "all", "label": f"All employees · {BASE_RATE:.0%}", "parent": "", "rate": BASE_RATE,
                          "n": len(Y), "left": Y.sum()}])  # explicit root: without it plotly paints a dark frame
    nodes = pd.concat([root, cells])
    mid = BASE_RATE / 0.4  # the average sits at the neutral midpoint: greener below, redder above
    fig = go.Figure(go.Treemap(
        ids=nodes["id"], labels=nodes["label"], parents=nodes["parent"], values=nodes["n"], branchvalues="total",
        customdata=nodes[["rate", "left", "n"]], texttemplate="<b>%{label}</b><br>%{customdata[0]:.0%}",
        hovertemplate="<b>%{label}</b><br>%{customdata[0]:.0%} left (%{customdata[1]} of %{customdata[2]})"
                      "<extra></extra>",
        marker=dict(colors=nodes["rate"], colorscale=[[0, "#1a7f37"], [mid, "#f1f1f1"], [(1 + mid) / 2, "#ec4647"], [1, "#8f1d1e"]], cmin=0,
                    cmax=0.4, line=dict(width=2, color="white"),
                    colorbar=dict(title="Left", tickvals=[0, BASE_RATE, 0.4], thickness=12,
                                  ticktext=["0 %", f"average {BASE_RATE:.0%}", "40 %+"])),
        textfont_size=14, textposition="middle center", pathbar_visible=False, tiling_pad=2, root_color="white"))
    return style(fig, height=520)


# ---------------------------------------------------------------- 3 trust
CUTS = np.round(np.arange(0.05, 0.71, 0.01), 2)


def counts(cut):
    """Returns caught leavers, missed leavers, false alarms and correctly ignored stayers at one cut-off."""
    flagged, left = RISK >= cut, Y_TEST == 1
    return (flagged & left).sum(), (~flagged & left).sum(), (flagged & ~left).sum(), (~flagged & ~left).sum()


CURVE = pd.DataFrame([counts(c) for c in CUTS], index=CUTS, columns=["tp", "fn", "fp", "tn"])
CURVE["recall"] = CURVE["tp"] / (CURVE["tp"] + CURVE["fn"])
CURVE["precision"] = CURVE["tp"] / (CURVE["tp"] + CURVE["fp"]).replace(0, np.nan)


def trust_fig(cut):
    """Line chart: share of leavers caught and hit rate among flagged, for every cut-off."""
    fig = go.Figure()
    for column, name, color in [("recall", "Leavers caught", BLUE), ("precision", "Flagged who really left", ORANGE)]:
        fig.add_scatter(x=CURVE.index, y=CURVE[column], name=name, mode="lines", line=dict(color=color, width=2),
                        hovertemplate="%{y:.0%}")
    fig.add_vline(x=cut, line_color=INK, line_width=1, annotation_text="your cut-off")
    fig.add_hline(y=Y_TEST.mean(), line_dash="dash", line_color=MUTED, line_width=1,
                  annotation_text="by chance", annotation_position="bottom right",
                  annotation_font_color=MUTED)
    return style(fig, height=300, hovermode="x unified", yaxis=dict(tickformat=".0%", range=[0, 1.02]),
                 xaxis=dict(tickformat=".0%", title="Cut-off"))


# ---------------------------------------------------------------- 4 global drivers
def drivers_fig(k=12):
    """Bar chart: average impact of the top features on the predicted risk, levers highlighted."""
    top = IMPORTANCE.head(k)
    fig = go.Figure()
    for is_lever, name, color in [(True, "Can be changed", BLUE), (False, "Given", GRAY)]:
        part = top[[(f in LEVERS) == is_lever for f in top.index]]
        fig.add_bar(x=part.values, y=[readable(f) for f in part.index], orientation="h", name=name,
                    marker_color=color, marker_cornerradius=4,
                    hovertemplate="<b>%{y}</b><br>shifts the risk by %{x:.1f} pp on average<extra></extra>")
    order = [readable(f) for f in top.index[::-1]]
    return style(fig, height=420, barmode="stack", xaxis_title="Effect on risk (percentage points)",
                 yaxis=dict(categoryorder="array", categoryarray=order))


def tipping_fig(feature):
    """Bar chart: average SHAP value per value range of one feature."""
    bands = pd.cut(RAW.loc[X_TEST.index, feature], BANDS[feature])
    t = (SHAP[feature] * 100).groupby(bands, observed=True).agg(["mean", "size"])
    labels = [f"{b.left + 1:,.0f}–{b.right:,.0f}" for b in t.index]
    fig = go.Figure(go.Bar(x=labels, y=t["mean"], marker_color=[ORANGE if v > 0 else BLUE for v in t["mean"]],
                           marker_cornerradius=4, customdata=t["size"],
                           hovertemplate="<b>%{x}</b><br>%{y:+.1f} pp on the risk<br>%{customdata} employees"
                                         "<extra></extra>"))
    fig.add_hline(y=0, line_color="#c2c2c2", line_width=1)
    return style(fig, height=360, xaxis=dict(type="category", title=readable(feature)),
                 yaxis_title="Effect on risk (pp)")


# ---------------------------------------------------------------- 5 + 6 employee explorer & what-if
def scenario(employee, overtime, raise_pct, stock, travel):
    """Returns the model row of one employee with the what-if changes applied."""
    row = X_TEST.loc[[employee]].astype(float)
    row["OverTime"], row["StockOptionLevel"], row["BusinessTravel"] = overtime, stock, travel
    row["MonthlyIncome"] *= 1 + raise_pct / 100
    return row


def local_fig(row, values, k=8):
    """Bar chart: the features that push this employee's risk up or down the most."""
    top = values.reindex(values.abs().sort_values(ascending=False).index)
    bars = pd.concat([top.head(k).rename(lambda f: f"{readable(f)}: {shown(row, f)}"),
                      pd.Series({"All other features": top.iloc[k:].sum()})])[::-1] * 100
    fig = go.Figure(go.Bar(x=bars.values, y=bars.index, orientation="h", marker_cornerradius=4,
                           marker_color=[ORANGE if v > 0 else BLUE for v in bars.values],
                           text=[f"{v:+.1f}" for v in bars.values], textposition="outside", cliponaxis=False,
                           hovertemplate="<b>%{y}</b><br>%{x:+.1f} pp on the risk<extra></extra>"))
    fig.add_vline(x=0, line_color="#c2c2c2", line_width=1)
    return style(fig, height=380, xaxis_title="← lowers risk · raises risk →")


# ---------------------------------------------------------------- 7 levers
def lever_table():
    """Applies each lever to every employee it fits and reports how the model reacts."""
    levers = [("End overtime", X_TEST["OverTime"] == 1, lambda d: d.assign(OverTime=0)),
              ("Raise salary by 10 %", X_TEST["OverTime"] >= 0, lambda d: d.assign(MonthlyIncome=d["MonthlyIncome"] * 1.1)),
              ("Grant stock options", X_TEST["StockOptionLevel"] == 0, lambda d: d.assign(StockOptionLevel=1)),
              ("Less business travel", X_TEST["BusinessTravel"] == 2, lambda d: d.assign(BusinessTravel=1))]
    rows = []
    for name, fits, change in levers:
        before = RISK[fits]
        after = RF.predict_proba(change(X_TEST[fits]))[:, 1]
        rows.append(html.Tr([html.Td(name), html.Td(f"{fits.sum()} of {len(X_TEST)}"),
                             html.Td(f"{before.mean():.0%} → {after.mean():.0%}"),
                             html.Td(f"{(after.mean() - before.mean()) * 100:+.1f} pp", className="strong"),
                             html.Td(f"{(before >= TUNED).sum()} → {(after >= TUNED).sum()}")]))
    head = ["Measure", "Employees", "Average risk", "Change", "Flagged"]
    return html.Table([html.Thead(html.Tr([html.Th(h) for h in head])), html.Tbody(rows)])


# ---------------------------------------------------------------- layout
TP, FN, FP, _ = counts(TUNED)
EMPLOYEES = [{"label": f"Employee {i} · risk {r:.0%}", "value": i} for i, r in RISK.sort_values(ascending=False).items()]
CAVEAT = "Model estimate, not a guarantee."

app = Dash(__name__, title="Attrition PoC · Passau MedCo")
app.index_string = """<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}<style>
body{margin:0;background:#f9f9f9;color:#191919;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 24px 60px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:18px;margin:0 0 4px} h3{font-size:14px;margin:0 0 8px}
.sub,.answer,.note,.lab,.caveat,th{color:#666}
.answer{margin:0 0 16px;max-width:860px} .caveat{font-size:13px;margin:12px 0 0}
section{background:#fff;border:1px solid #e7e7e7;border-radius:10px;padding:22px 24px;margin-top:18px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:18px}
.tile{background:#fff;border:1px solid #e7e7e7;border-radius:10px;padding:16px 18px}
section .tile{background:#f9f9f9}
.num{font-size:30px;font-weight:650;line-height:1.1} .lab{font-size:13px;margin-top:6px} .note{font-size:12px;margin-top:2px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:28px} .controls{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;margin:16px 0}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
button{border:1px solid #c2c2c2;background:#fff;border-radius:6px;padding:7px 12px;font-size:13px;cursor:pointer}
button:hover{background:#f1f1f1} label{font-size:13px;margin-right:10px}
table{border-collapse:collapse;width:100%;font-size:14px} th,td{text-align:left;padding:9px 10px;border-bottom:1px solid #e7e7e7}
th{font-weight:500;font-size:13px} .strong{font-weight:650}
details{margin-top:14px;font-size:13px;color:#666} summary{cursor:pointer}
@media(max-width:900px){.tiles,.cols,.controls{grid-template-columns:1fr 1fr}}
</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""

app.layout = html.Main([
    html.H1("Why do employees leave?"),
    html.Div("Attrition PoC · Passau MedCo · 1,470 employees", className="sub"),
    html.Div([
        tile(f"{BASE_RATE:.0%}", "left the company"),
        tile(f"{TP / (TP + FP):.0%}", "of flagged employees really left", f"{Y_TEST.mean():.0%} by chance"),
        tile(f"{TP} of {TP + FN}", "leavers spotted in advance"),
        tile(readable(IMPORTANCE.index[0]), "biggest driver"),
    ], className="tiles"),

    section("Who leaves?",
            html.Div([html.Label("Split by"),
                      dcc.Dropdown([{"label": readable(g), "value": g} for g in GROUPINGS], "JobRole", id="group",
                                   clearable=False, style={"width": 260})], className="row"),
            dcc.Graph(id="workforce", config=PLOT)),

    section("Can we see it coming?",
            html.P("Move the cut-off: lower catches more leavers, but causes more false alarms.", className="answer"),
            dcc.Slider(0.05, 0.70, 0.01, value=round(TUNED, 2), id="cut", tooltip={"always_visible": True},
                       marks={c / 100: f"{c} %" for c in range(10, 71, 10)}),
            html.Div(id="trust-tiles"),
            dcc.Graph(id="trust", config=PLOT)),

    section("What drives it?",
            html.Div([
                html.Div([html.H3("Biggest drivers"), dcc.Graph(figure=drivers_fig(), config=PLOT)]),
                html.Div([html.Div([html.H3("Where does the risk tip?", style={"margin": 0}),
                                    dcc.Dropdown([{"label": readable(f), "value": f} for f in BANDS], "MonthlyIncome",
                                                 id="band", clearable=False, style={"width": 240})], className="row"),
                          dcc.Graph(id="tipping", config=PLOT)]),
            ], className="cols")),

    section("Why this person, and what if?",
            html.Div([dcc.Dropdown(EMPLOYEES, CASES["Clear leaver"], id="employee", clearable=False, style={"width": 280}),
                      *[html.Button(name, id=f"case-{i}") for i, name in enumerate(CASES)]], className="row"),
            html.Div(id="verdict"),
            html.Div([
                html.Div([html.H3("Overtime"), dcc.RadioItems([{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                                                             id="w-overtime", inline=True)]),
                html.Div([html.H3("Salary raise"), dcc.Slider(0, 30, 5, value=0, id="w-raise",
                                                              marks={p: f"{p} %" for p in range(0, 31, 10)})]),
                html.Div([html.H3("Stock options"), dcc.RadioItems([0, 1, 2, 3], id="w-stock", inline=True)]),
                html.Div([html.H3("Business travel"), dcc.RadioItems([{"label": v, "value": k} for k, v in TRAVEL_LABEL.items()],
                                                                    id="w-travel", inline=True)]),
            ], className="controls"),
            dcc.Graph(id="local", config=PLOT),
            html.P(CAVEAT, className="caveat")),

    section("What helps most?", lever_table(), html.P(CAVEAT, className="caveat")),
])


# ---------------------------------------------------------------- callbacks
app.callback(Output("workforce", "figure"), Input("group", "value"))(workforce_fig)
app.callback(Output("tipping", "figure"), Input("band", "value"))(tipping_fig)


@app.callback(Output("trust-tiles", "children"), Output("trust", "figure"), Input("cut", "value"))
def update_trust(cut):
    tp, fn, fp, tn = counts(cut)
    flagged = tp + fp
    hit = f"{tp / flagged:.0%}" if flagged else "–"
    tiles = html.Div([tile(str(flagged), "flagged", f"of {len(Y_TEST)} employees"),
                      tile(str(tp), "really left", f"{hit} of flagged"),
                      tile(str(fn), "leavers missed"),
                      tile(str(fp), "false alarms")], className="tiles")
    return tiles, trust_fig(cut)


@app.callback(Output("employee", "value"), [Input(f"case-{i}", "n_clicks") for i in range(len(CASES))],
              prevent_initial_call=True)
def pick_case(*_):
    return list(CASES.values())[int(ctx.triggered_id.split("-")[1])]


@app.callback(Output("w-overtime", "value"), Output("w-raise", "value"), Output("w-stock", "value"),
              Output("w-travel", "value"), Input("employee", "value"))
def reset_levers(employee):
    row = X_TEST.loc[employee]
    return int(row["OverTime"]), 0, int(row["StockOptionLevel"]), int(row["BusinessTravel"])


@app.callback(Output("verdict", "children"), Output("local", "figure"), Input("employee", "value"),
              Input("w-overtime", "value"), Input("w-raise", "value"), Input("w-stock", "value"),
              Input("w-travel", "value"))
def update_employee(employee, overtime, raise_pct, stock, travel):
    actual = X_TEST.loc[employee]
    if ctx.triggered_id == "employee" or overtime is None:  # levers still hold the previous employee's values
        overtime, raise_pct, stock, travel = actual["OverTime"], 0, actual["StockOptionLevel"], actual["BusinessTravel"]
    row = scenario(employee, overtime, raise_pct, stock, travel)
    _, values = explain(row)
    values, before, after = values.iloc[0], RISK[employee], float(RF.predict_proba(row)[0, 1])
    changed = abs(after - before) > 1e-9
    verdict = html.Div([
        html.Div([tile(f"{before:.0%}", "risk of leaving", f"average {BASE:.0%}"),
                  tile(f"{after:.0%}" if changed else "–", "risk after changes",
                       f"{(after - before) * 100:+.1f} pp" if changed else ""),
                  tile("Flagged" if after >= TUNED else "Not flagged", "model says"),
                  tile("Left" if Y_TEST[employee] else "Stayed", "what really happened")],
                 className="tiles", style={"marginTop": 0})])
    return verdict, local_fig(row.iloc[0], values)


if __name__ == "__main__":
    app.run(debug=False)
