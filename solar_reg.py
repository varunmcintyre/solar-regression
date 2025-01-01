import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

app = Dash(__name__)

app.layout = html.Div([
    # Add title
    html.H1("Electricity Generation Projections by Source through 2030", style={"text-align": "center"}),
    
    dcc.Graph(id="reg_graph", style={'width': '100vw', 'height': '70vh'}),

    html.P("Select Electricity Source"),
    dcc.Dropdown(id="source_choice", options=["Solar", "Wind", "Coal"], value="Solar", style={'width': '10vw'})
])

def prep_data(df, source):
    # create new columns with just year, month, and reformatted date
    df["Year"] = df["YYYYMM"] / 100
    df["Year"] = df["Year"].apply(lambda x: int(x))
    df["Month"] = df["YYYYMM"] % 100
    df["Date"] = df["YYYYMM"].apply(lambda x: str(x)[4:]+"/"+str(x)[:4])

    # take out year totals (Month = 13), and keep only 2010 through 2023, the last complete year
    df = df.loc[df["Month"] != 13]
    df = df.loc[df["Year"] >= 2010]
    df = df.loc[df["Year"] <= 2023]

    # keep only chosen source and remove unavailable data
    df = df.loc[df["MSN"] == source]
    df = df.loc[df["Value"] != "Not Available"]
    
    # keep necessary columns
    df = df[["Value", "Year", "Month", "Date"]]

    # make sure all values in Value are floats
    df["Value"] = df["Value"].apply(lambda x: float(x))
    
    return df


def train_regression_model(df):
    X = df[["Year", "Month"]]
    y = df["Value"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    poly = PolynomialFeatures(degree=4)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_poly, y_train)

    y_pred = lr.predict(X_test_poly)

    r2 = r2_score(y_test, y_pred)
    print("R2:",round(r2, 3))

    return lr, poly


def make_projections(lr, poly):
    months = []
    years = []

    for i in range(7):
        for j in range(1,13):
            months.append(j)
            years.append(i+2024)

    X_proj = pd.DataFrame({"Year":years, "Month":months})
    
    X_proj_poly = poly.transform(X_proj)
    y_proj = lr.predict(X_proj_poly)

    return X_proj, y_proj


def add_projections(df, X_proj, y_proj):

    X_proj["Date"] = X_proj["Month"].apply(lambda x: str(x)) + "/" + X_proj["Year"].apply(lambda x: str(x))
    y_proj = pd.DataFrame({"Value":y_proj})
    X_proj["Value"] = y_proj["Value"]

    final_df = pd.concat([df, X_proj], ignore_index=True)

    return final_df


@app.callback(
    Output("reg_graph", "figure"),
    Input("source_choice", "value"),
)
def plot(source_choice):
    df = pd.read_csv("MER_T07_02A.csv")

    source_dict = {"Solar":"SOETPUS", "Wind":"WYETPUS", "Coal":"CLETPUS"}
    df = prep_data(df, source_dict[source_choice])

    lr, poly = train_regression_model(df)
    X_proj, y_proj = make_projections(lr, poly)
    final_df = add_projections(df, X_proj, y_proj)

    reg_fig = go.Figure()

    trace = go.Scatter(x=final_df["Date"], y=final_df["Value"], mode="lines")
    reg_fig.add_traces(trace)

    reg_fig.update_layout(
        title=f"Electricity Generated from {source_choice}",
        yaxis_title="Million Killowatthours",
        xaxis_title="Month"
    )

    return reg_fig



if __name__ == "__main__":
    app.run_server(debug=True)


'''
- separate plot function from generating projections
- make a bunch of projections and choose the one with the highest R2
- display the R2
'''

# test