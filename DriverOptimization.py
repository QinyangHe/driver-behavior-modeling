import gurobipy as gp
from gurobipy import GRB
import numpy as np

gamma_1 = 0.2
gamma_2 = 0.2
gamma_3 = 0.2
lambda_ = 0.0001

def fit_parameters(claim_frequency, claim_amount, discount, time_limit=10):
    """
    Solve the sensitivity parameter fitting problem.
    If time limit is reached, prioritize finding a feasible solution.
    
    Parameters:
    - claim_frequency: List of claim frequency data
    - claim_amount: List of claim amount data
    - discount: List of discount data
    - lambda_: Weight factor for claim amount difference in objective
    - time_limit: Time limit in seconds for the optimization
    
    Returns:
    - Dictionary containing fitted parameters and values
    """
    T = len(claim_frequency)
    model = gp.Model("SensitivityParameterFitting")
    
    # Set parameters to prioritize feasibility when time limit is reached
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)
    model.setParam('MIPGap', 0.1)
    model.setParam('FeasibilityTol', 1e-6)
    model.setParam('IntFeasTol', 1e-6)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    
    # Variables
    #x = {t: model.addVar(name=f"x_{t}") for t in range(T)}
    d = {t: model.addVar(lb=0, name=f"d_{t}") for t in range(T)}
    d_diff = {t: model.addVar(name=f"d_diff_{t}") for t in range(T)}
    y = {t: model.addVar(lb=0, name=f"y_{t}") for t in range(T)}
    y_diff = {t: model.addVar(name=f"y_diff_{t}") for t in range(T)}
    
    c = model.addVar(lb=0, name="c")
    D = model.addVar(lb=0, name="D")
    Y = model.addVar(lb=0, name="Y")
    #alpha_1 = model.addVar(name="alpha_1")
    beta_1 = model.addVar(name="beta_1")
    theta_1 = model.addVar(name="theta_1")
    #alpha_2 = model.addVar(name="alpha_2")
    beta_2 = model.addVar(name="beta_2")
    theta_2 = model.addVar(name="theta_2")
    
    # Initial conditions
    model.addConstr(d[0] == D)
    model.addConstr(y[0] == Y)
    model.addConstr(c == 0)
    #model.addConstr(x[0] == 0)
    
    # Dynamics constraints
    for t in range(T - 1):
        # model.addConstr(
        #     x[t + 1] == x[t] + alpha_1 * (discount[t + 1] - discount[t]) + alpha_2 * (discount[t + 1] - c),
        #     name=f"x_dyna_{t+1}"
        # )

        model.addConstr(
            d[t + 1] == beta_1 * d[t] + beta_2 * (d[t] - D) + D,
            name=f"d_dyna_{t+1}"
        )

        model.addConstr(
            y[t + 1] == theta_1 * y[t] + theta_2 * (y[t] - Y) + Y,
            name=f"y_dyna_{t+1}"
        )
    
    # Objective function
    obj = gp.LinExpr()
    for t in range(T):
        model.addConstr(d_diff[t] >= d[t] - claim_frequency[t])
        model.addConstr(d_diff[t] >= claim_frequency[t] - d[t])
        model.addConstr(y_diff[t] >= y[t] - claim_amount[t])
        model.addConstr(y_diff[t] >= claim_amount[t] - y[t])
        obj += (d_diff[t] + lambda_ * y_diff[t])
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Optimize in two phases if needed
    model.optimize()
    
    # Check if time limit was reached and solution is not optimal
    if model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
        print("Time limit reached but feasible solution found.")
        # We already have a feasible solution, extract it
    elif model.Status == GRB.TIME_LIMIT and model.SolCount == 0:
        print("Time limit reached with no feasible solution. Trying to find a feasible solution...")
        # Change focus to finding any feasible solution
        model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
        model.setParam('TimeLimit', time_limit)  # Give it another time_limit seconds
        
        # Create a feasibility problem by relaxing the objective
        model.setObjective(0, GRB.MINIMIZE)  # Zero objective to focus purely on feasibility
        model.optimize()
    
    # Check results
    if model.SolCount > 0:
        print(f"Solution found with status: {model.Status}")
        
        # Extract results
        results = {
            #'x': {t: x[t].X for t in range(T)},
            'd': {t: d[t].X for t in range(T)},
            'y': {t: y[t].X for t in range(T)},
            "c": c.X,
            "D": D.X,
            "Y": Y.X,
            #"alpha_1": alpha_1.X,
            #"alpha_2": alpha_2.X,
            "beta_1": beta_1.X,
            "beta_2": beta_2.X,
            "theta_1": theta_1.X,
            "theta_2": theta_2.X,
            'objective_value': model.objVal,
            'solution_status': model.Status,
            'is_optimal': model.Status == GRB.OPTIMAL
        }
        print(results)
        return results
    else:
        print(f"No feasible solution found. Status code: {model.Status}")
        if model.Status == GRB.INFEASIBLE:
            model.computeIIS()
            for c in model.getConstrs():
                if c.IISConstr:
                    print(f"Constraint {c.ConstrName} contributes to infeasibility")
        return None

def compute_discount(results_list, discount_matrix, premium):
    x_t_list = []
    d_t_list = []
    y_t_list = []
    c_list = []
    D_list = []
    Y_list = []
    #alpha_1_list = []
    #alpha_2_list = []
    beta_1_list = []
    beta_2_list = []
    theta_1_list = []
    theta_2_list = []
    num_drivers = len(results_list)
    for results in results_list:
        #x = {t: results['x'][t] for t in results['x']}
        d = {t: results['d'][t] for t in results['d']}
        y = {t: results['y'][t] for t in results['y']}
        
        # Get the final values at time T
        T = len(d) - 1
        #x_t_list.append(x[T])
        d_t_list.append(d[T])
        y_t_list.append(y[T])
        
        # Append scalar parameters to their respective lists
        c_list.append(results["c"])
        D_list.append(results["D"])
        Y_list.append(results["Y"])
        #alpha_1_list.append(results["alpha_1"])
        #alpha_2_list.append(results["alpha_2"])
        beta_1_list.append(results["beta_1"])
        beta_2_list.append(results["beta_2"])
        theta_1_list.append(results["theta_1"])
        theta_2_list.append(results["theta_2"])
    

    model = gp.Model("ComputeOptimalDiscount")
    #x_plus = {}
    d_plus = {}
    y_plus = {}
    discount = {}
    total_premium = {}
    expected_cost = {}
    print(D_list)
    print(Y_list)
    #print("x_t_list: " + str(x_t_list))
    #print("alpha_1" + str(alpha_1_list))
    #print("alpha_2" + str(alpha_2_list))
    print("beta_1" + str(beta_1_list))
    print("beta_2" + str(beta_2_list))
    print("theta_1" + str(theta_1_list))
    print("theta_2" + str(theta_2_list))
    for i in range(num_drivers):
        #x_plus[i] = model.addVar()
        y_plus[i] = model.addVar()
        d_plus[i] = model.addVar()
        total_premium[i] = model.addVar()
        expected_cost[i] = model.addVar()
        discount[i] = model.addVar(lb = 0)
        past_discount = discount_matrix[i, :]
        # model.addConstr(
        #         x_plus[i] == x_t_list[i] + alpha_1_list[i] * (discount[i] - past_discount[-1]) + alpha_2_list[i] * (discount[i] - c_list[i])
        #     )

        model.addConstr(
                d_plus[i] == beta_1_list[i] * d_t_list[i] * discount[i] + beta_2_list[i] * (d_t_list[i] - D_list[i]) + D_list[i]
            )
        
        model.addConstr(
                y_plus[i] == theta_1_list[i] * y_t_list[i] * discount[i] + theta_2_list[i] * (y_t_list[i] - Y_list[i]) + Y_list[i]
            )
        model.addQConstr(
            expected_cost[i] == d_plus[i] * y_plus[i]
        )
        # print(premium[i])
        # print(1 - discount[i])
        # model.addConstr(
        #     total_premium[i] ==  premium[i][0] * discount[i]
        # )

    obj = gp.QuadExpr()
    for i in range(num_drivers):
        # print(premium[i] * (1 - discount[i]))
        # print(d_plus[i] * y_plus[i])
        
        obj += expected_cost[i] + premium[i][0] * discount[i]

    model.setObjective(obj, GRB.MINIMIZE)
    model.setParam("NumericFocus", 1)
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        results = {
            'discount': [discount[i].X for i in range(num_drivers)],
            #'x_plus': [x_plus[i].X for i in range(num_drivers)],
            'd_plus': [d_plus[i].X for i in range(num_drivers)],
            'y_plus': [y_plus[i].X for i in range(num_drivers)],
            'objective_value': model.objVal,
            'status': 'optimal'
        }
        return results
    else:
        print(f"Optimization failed with status code {model.status}")
        return None