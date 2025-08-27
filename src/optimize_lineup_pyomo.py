import pandas as pd
import pyomo.environ as pyo
from pathlib import Path
import sys
from pyomo.contrib.appsi.solvers import Highs
from pyomo.contrib.appsi.base import TerminationCondition as TC

print("Using Python:", sys.executable) # making sure in right environment

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "data" / "players_week1.csv" # get csv file path
assert Path(CSV_PATH).exists(), f"CSV not found at {CSV_PATH}" # make sure CSV file is found
print("Reading:", CSV_PATH)

df = pd.read_csv(CSV_PATH) # create df
#print(df.head())


# Pyomo IP model
m = pyo.ConcreteModel() # use concrete model 

#define model set of players using index 0-24
players = list(df.index)
m.I = pyo.Set(initialize = players)

#define model parameters
salary = df['salary'].to_dict()
m.salary = pyo.Param(m.I, initialize = salary)

proj_points = df['proj_points'].to_dict()
m.proj_points = pyo.Param(m.I, initialize = proj_points)

team = df['team'].to_dict() # using regular dictionary isntead of parameter for simplicity
#m.team = pyo.Param(m.I, initialize = team, domain = pyo.Any)

position = df['position'].to_dict() # using regular dictionary isntead of parameter for simplicity
#m.position = pyo.Param(m.I, initialize = position, domain = pyo.Any)

cap = 60_000 # salary cap

#define decision variable 
m.x = pyo.Var(m.I, domain = pyo.Binary)

#define functions
def obj_max_points(m):
    return sum(m.proj_points[i] * m.x[i] for i in m.I) # total sum of maximum possible points
m.Obj = pyo.Objective(rule = obj_max_points, sense = pyo.maximize)

def salary_cap_constraint_rule(m):
    return sum(m.salary[i] * m.x[i] for i in m.I) <= cap # players must add up to be under salary cap
m.c1 = pyo.Constraint(rule = salary_cap_constraint_rule)

# create a list for each position
QB = [i for i in m.I if position[i] == 'QB'] 
RB = [i for i in m.I if position[i] == 'RB']
WR = [i for i in m.I if position[i] == 'WR']
TE = [i for i in m.I if position[i] == 'TE']
DST = [i for i in m.I if position[i] == 'DST']

# set constraint on number of players per team, no bench players for now
m.one_QB = pyo.Constraint(expr = sum(m.x[i] for i in QB) == 1)
m.one_DST = pyo.Constraint(expr = sum(m.x[i] for i in DST) == 1)
m.min_RB = pyo.Constraint(expr = sum(m.x[i] for i in RB) >= 2)
m.min_WR = pyo.Constraint(expr = sum(m.x[i] for i in WR) >= 3)
m.min_TE = pyo.Constraint(expr = sum(m.x[i] for i in TE) >= 1)
m.flex = pyo.Constraint(expr = sum(m.x[i] for i in RB+WR+TE) == 7)

# make sure I have at least 1 player from each position
assert all(len(S) > 0 for S in (QB, RB, WR, TE, DST)), "Missing players in a position"

#check what the minimum salary is to ensure we have at least one feasible solution less than salary cap
min_possible = min(salary[i] for i in QB) \
             + min(salary[i] for i in DST) \
             + sum(sorted(salary[i] for i in RB)[:2]) \
             + sum(sorted(salary[i] for i in WR)[:3]) \
             + sum(sorted(salary[i] for i in TE)[:1]) \
             + min(sorted(salary[i] for i in (RB+WR+TE)))
print("Min possible salary:", min_possible)

# solver High
opt = Highs()
result = opt.solve(m)
print("Status:", result.termination_condition)
# if unbounded/infeasible 
if result.termination_condition != TC.optimal:
    raise RuntimeError(f"Solve failed: {result.termination_condition}")

#print(result)
#m.display()

#create list for each player that was selected, 1 means selected 0 means not selected
chosen_idx = [i for i in m.I if pyo.value(m.x[i]) > 0.5]
lineup = df.loc[chosen_idx, ["name","position","team","salary","proj_points"]].sort_values(["position","proj_points"], 
                                                                                           ascending=[True, False])
# print out the team
print("\n--- Optimal lineup ---")
print(lineup.to_string(index=False))
print("Total salary:", int(lineup["salary"].sum()))
print("Projected pts:", round(lineup["proj_points"].sum(), 2))