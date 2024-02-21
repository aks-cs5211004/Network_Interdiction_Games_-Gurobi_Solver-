import gurobipy as gp

model = gp.Model()

    # Step 2: Add a dummy variable and constraint
x = model.addVar(lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name="x")
model.addConstr(x <= 1, name="constraint")

    # Step 3: Update the model
model.update()

    # Step 4: Optimize the model
model.optimize()

    # Step 5: Check if the model is successfully solved
if model.status == gp.GRB.OPTIMAL:
        print("Gurobi is working correctly!")
        print(f"Optimal solution: x = {x.x}")
else:
        print("Gurobi encountered an issue while solving the model.")
        print(f"Model status: {model.status}")
