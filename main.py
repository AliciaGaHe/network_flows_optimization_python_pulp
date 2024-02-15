from pulp import *
import pandas as pd
import re


def solve_problem_using_pulp(data_file):
    """solve the problem using pulp and print results"""
    ### Load data ###

    # Open json file with data
    f = open(data_file)

    # Returns the data as a dictionary
    data = json.load(f)

    # Close the file
    f.close()

    ### Create the objects that the optimization model needs ###

    # Lists with sources and customers
    sSources = data["sSources"]
    sCustomers = data["sCustomers"]

    # Production limit for each source and demand for each customer
    pSourceProduction = data["pSourceProduction"]
    pCustomerDemand = data["pCustomerDemand"]

    # Transportation costs for each source and customer
    pTransportationCosts = {tuple(item["index"]): item["value"] for item in data["pTransportationCosts"]}

    # List with the available combinations of sources and customers
    sSources_Customers = [tuple(item["index"]) for item in data["pTransportationCosts"]]

    # Quantity that is mandatory to move between each source and customer
    pFixedTransportation = {tuple(item["index"]): item["value"] for item in data["pFixedTransportation"]}

    ### Create the model ###

    # Instantiate the model class
    model = LpProblem("Network_flows_optimization", LpMinimize)

    # Create the decision variables
    vQuantityExchanged = LpVariable.dicts(
        "quantity_in_tons",
        ((s, c) for s in sSources for c in sCustomers if (s, c) in sSources_Customers),
        lowBound=0,
        cat='Continuous')

    # Create objective function
    model += lpSum([
        pTransportationCosts[s, c] * vQuantityExchanged[s, c]
        for s in sSources for c in sCustomers if (s, c) in sSources_Customers
    ])

    # Create the constraints

    # Production limit for each source
    for s in sSources:
        model += (
            lpSum([vQuantityExchanged[s, c] for c in sCustomers if (s, c) in sSources_Customers])
            <= pSourceProduction[s],
            "c01_production_%s" % s
        )

    # Demand limit for each customer
    for c in sCustomers:
        model += (
            lpSum([vQuantityExchanged[s, c] for s in sSources if (s, c) in sSources_Customers])
            >= pCustomerDemand[c],
            "c02_demand_%s" % c
        )

    # Quantity that is mandatory to move between each source and each customer
    for s in sSources:
        for c in sCustomers:
            if (s, c) in sSources_Customers and pFixedTransportation[s, c]:
                model += (
                        vQuantityExchanged[s, c] == pFixedTransportation[s, c],
                        "c03_fixed_%s_%s" % (s, c)
                )

    ### Export the formulation to a file ###
    # model.writeLP("formulation.pl")

    ### Solve the model ###

    # If we want to review the available solvers, we can use...
    # solver_list = listSolvers(onlyAvailable=True)
    # print(solver_list)

    # Select the solver
    solver = getSolver('PULP_CBC_CMD')

    # We can change some solver options before solving (such as the maximum time limit),
    # but it is not relevant in this problem because it is easy to solve
    # solver = getSolver('PULP_CBC_CMD', timeLimit=10)

    # Solve the model using the chosen solver
    model.solve(solver)

    ### Print results ###
    print("\n")
    print("Solver status: ", LpStatus[model.status], "\n")

    print("Total transportation cost: ", value(model.objective), "\n")

    print("Quantity exchanged between sources and customers:")
    dict_quantity_sources_customers = {
        (s, c): value(vQuantityExchanged[s, c])
        for s in sSources for c in sCustomers if (s, c) in sSources_Customers and value(vQuantityExchanged[s, c]) > 0
        }
    df_quantity_sources_customers = pd.DataFrame(dict_quantity_sources_customers.values(),
                                                 index=pd.MultiIndex.from_tuples(
                                                     dict_quantity_sources_customers.keys()),
                                                 columns=['Quantity']
                                                 ).reset_index(names=['Source', 'Customer'])
    print(df_quantity_sources_customers)
    print("\n")

    print("Sensibility analysis - constraints:")
    list_sensibility_analysis_constraints = [
        {'Constraint': name, 'Slack': c.slack, 'Shadow price': c.pi}
        for name, c in model.constraints.items()]
    df_sensibility_analysis_constraints = pd.DataFrame(list_sensibility_analysis_constraints)
    print(df_sensibility_analysis_constraints)
    print("\n")

    # Conclusions
    df_sensibility_analysis_constraints[
        df_sensibility_analysis_constraints['Slack'] == 0
        ].apply(
        lambda row:
        print_conclusions_constraints_sensibility_analysis(row['Constraint'], row['Shadow price'], sSources), axis=1)
    print("\n")

    print("Sensibility analysis - variables:")
    list_sensibility_analysis_variables = [
        {'Variable': v.name, 'Value': v.varValue, 'Reduced cost': v.dj}
        for v in model.variables()]
    df_sensibility_analysis_variables = pd.DataFrame(list_sensibility_analysis_variables)
    print(df_sensibility_analysis_variables)
    print("\n")

    # Conclusions
    df_sensibility_analysis_variables[
        df_sensibility_analysis_variables['Value'] == 0
        ].apply(
        lambda row:
        print_conclusions_variables_sensibility_analysis(row['Variable'], row['Reduced cost']), axis=1)
    print("\n")


def print_conclusions_constraints_sensibility_analysis(constraint_name, shadow_price, sources):
    """print conclusions of the constraints sensibility analysis"""
    # Find the constraint number and the constraint location using the constraint name
    constraint_number = int(str(constraint_name)[2:3])
    location = str(constraint_name)[-3:]
    if constraint_number <= 2 and location in sources:
        if shadow_price < 0:
            print("The total transportation cost would be reduced by",
                  abs(shadow_price),
                  "euros for each additional ton available in", location)
        elif shadow_price > 0:
            print("The total transportation cost would be increased in",
                  shadow_price,
                  "euros for each additional ton available in", location)
        else:
            print("The total transportation cost would remain equal for each additional ton available in", location)
    elif constraint_number <= 2:
        if shadow_price < 0:
            print("The total transportation cost would be reduced by",
                  abs(shadow_price),
                  "euros for each additional ton supply at", location)
        elif shadow_price > 0:
            print("The total transportation cost would be increased in",
                  shadow_price,
                  "euros for each additional ton supply at", location)
        else:
            print("The total transportation cost would remain equal for each additional ton supply at", location)


def print_conclusions_variables_sensibility_analysis(variable_name, reduced_cost):
    """print conclusions of the variables sensibility analysis"""
    # Find the source and the customer using the variable name
    list_source_customer = re.findall(r"(?<=[']).{3}(?=['])", variable_name)
    source = list_source_customer[0]
    customer = list_source_customer[1]
    if reduced_cost < 0:
        print("The total transportation cost would be reduced by",
              abs(reduced_cost),
              "euros for each ton supply from", source, "to", customer)
    elif reduced_cost > 0:
        print("The total cost would be increased in",
              reduced_cost,
              "euros for each ton supply from", source, "to", customer)
    else:
        print("The total transportation cost would remain equal for each ton supply from", source, "to", customer)


# Solve some transportation problems

# Base case
solve_problem_using_pulp("./data/data_0.json")

# Sensibility analysis - sources
# Using the base case, we move one ton of supply capacity from Gou to Arn and
# the objetive function improves in 0.2 euros (shadow price for Arn in the base case)
# solve_problem_using_pulp("./data/data_1.json")

# Sensibility analysis - customers - 1
# Using the base case, we increase the demand in Lon in one ton, and
# we increase one ton of supply capacity in Gou (Gou is the only source for Lon).
# Then the objetive function gets worse in 2.5 euros (shadow price for Lon in the base case)
# solve_problem_using_pulp("./data/data_2.json")

# Sensibility analysis - customers - 2
# Using the base case, we increase the demand in Ber in one ton, and
# we increase one ton of supply capacity in Gou (Arn is the only source for Ber).
# Then the objetive function gets worse in 2.7 euros (shadow price for Ber in the base case)
# solve_problem_using_pulp("./data/data_3.json")

# Sensibility analysis - routes
# Using the base case, we fixed a transportation between Arn and Ams equal to 1 ton,
# using pFixedTransportation and c03_fixed_%s_%s.
# The objetive function gets worse in 0.6 euros (reduced cost for the transportation between Arn and Ams)
# solve_problem_using_pulp("./data/data_4.json")
