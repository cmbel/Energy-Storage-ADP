# Energy-Storage-ADP

Need assistance with python coding. Please make corrections or improvements as needed for the code to run smoothly and accurately.

Main Goal: Determine an optimal energy storage dispatch schedule (i.e. when to charge, discharge, or idle) to minimize a residential household's electricity bill costs.

Some sample data from 10 residential households is provided. All meter data has units of kilowatt-hours (kWh). Load (also sometimes called "demand") and solar generation is measured by utility meters. There are 48 measurements (or "intervals") made during a 24 hour day. This means the sample data is half-hourly data. The code should be adaptable and able to handle 15-minute data (96 intervals) or hourly data (24 intervals) as well.

All households have their own solar installed. The homeowners are interested in installing an electric battery (also sometimes called a "battery" or "energy storage"), but are unsure how to schedule the battery. This is where an optimization algorithm will be helpful. The optimization algorithm will consider the load (kWh) and solar generation (kWh) and determine the best battery action ("charge", "discharge", or "idle") during every interval of the day. The output will be an optimal policy that tells the homeowner during which hours to charge, discharge, or idle.

There are some constraints. See the accompanying "Model Equations and Constraints.docx" file for mathematical model and constraint equations. Also please see the "Projected battery adoption in the prosumer era.pdf" paper for the origin of the constraint equations.
  1. The optimization algorithm cannot tell the battery to exceed its maximum power ratings. For example, it cannot charge more than 5 kilowatts (kW) or discharge more than 5 kilowatts (kW).
     
  2. The optimization algorithm cannot tell the battery to exceed its state of charge. For example, the battery cannot charge more than 100% state of charge and cannot discharge less than 0%.
     
  3. The battery cannot hold more than its energy capacity rating. For example, the Total Energy Capacity of the battery is 15 kWh of energy.
     
  4. State of Charge (SOC) is defined as:  SOC = Stored Energy Capacity / Total Energy Capacity  , For example, a battery storing 7.5 kWh has a State of Charge of 7.5 / 15 = 0.5 or 50%
     
  5. The SOC will increase or decrease as the battery charges or discharges. When charging, SOC[t] = SOC[t-1] + Energy_charged / charge_efficiency
  
  6. When discharging, SOC[t] = SOC[t-1] - Energy_discharged * discharge efficiency
     
  7. When charging, power (kW) has a positive sign. When discharging, the power has a negative sign.
     
  8. The battery can only be charged with solar energy. No energy from the grid can charge the battery. Therefore, the battery can only charge when solar meter data > load meter data.
     
  10. The battery cannot discharge and export energy to the grid. Therefore, the battery can only discharge when the load meter data > solar meter data. 
  
  11. The battery also cannot discharge more than the difference of load - solar. The battery can only help to supply extra energy that the solar cannot provide.


Algorithm
Due to my limited computer resources, I cannot run the intensive Dynamic Programming solution using a backwards induction. Dynamic Programming using backwards induction must calculate the Value of taking every action for every state. Doing this is a high dimensional task. I need a more efficient way of running the optimization algorithm.

My paper proposes using a Monte Carlo based "Approximate Dynamic Programming" (ADP) algorithm that uses a "Double Pass" strategy (also sometimes called an "ADP Double Pass algorithm" or "Double Pass" method). For more information about the ADP Double Pass algorithm, please read the "Approximate Dynamic Programming by Practical Example.pdf". Specifically, please read this paper's Section 2.3 Approximate Dynamic Programming.

As part of an ADP Double Pass algorithm, the "Forward Pass" intends to do the following:
  1. Setting the Initial State: The state of charge (SOC) of the battery at the beginning of the episode is initialized to 0 (i.e., the battery starts empty).
  2. Action Selection: For each time interval t, the algorithm looks up the current policy to determine which action (charging or discharging the battery) should be taken given the current battery energy level (battery_energy[t]).
  3. Action Validation: After selecting an action, the algorithm checks if the chosen action is valid based on certain constraints like the battery's minimum and maximum SOC, maximum charge and discharge rates, etc. If the action doesn't adhere to these constraints, it gets modified or rejected.
  4. Cost Calculation: The algorithm calculates the electricity cost for the current time interval based on the load demand, solar generation, battery action, and the electricity price.
  5. State Update: The battery's energy state for the next time interval t+1 is updated based on the action taken, considering constraints and efficiency factors.
  6. Trajectory Recording: The algorithm records the current trajectory, which consists of the battery's energy level, the action taken, and the observed cost.

The "Backward Pass" intends to do the following:
  Value Function Update: 
  1. the algorithm uses the trajectory (previously recorded in the Forward Pass) to get the state b, action a, and observed cost c.
  2. The future cost (for any interval other than the final interval which has no future cost) is computed based on the stored future values, or it's set to zero if evaluating the final interval.
  3. The Value Function V[t, s] is then updated using the observed cost and the mean future value. This update adjusts the estimate of the value of being in state s at time t.
  
     Policy Improvement:
  4. Find the best action, a, that minimizes the expected future cost given the current state.
  5. For each possible action, the algorithm checks whether the action adheres to the constraints.
  6. The future state is estimated based on the action.
  7. An expected future cost is then computed using the observed cost and any available future values.
  8. If this expected future cost is less than the current value V[t, b], the policy is updated with the more optimal action. Specifically, the action for the state, b, at time t (policy[t, b]) is updated, and the value function is updated to the computed expected future cost.


Monte Carlo used for the Value Function Approximation. Please see the 

The Value Function for a given state, action, time (s,t,a) is defined as:

V(s,t) = min( Electricity Cost(s,t,a) + B * Sum( P(s',r | s, a) * V(s') ) , for all future states and actions s', a

  Where: P(s',r | s, a) is the probabilities of transitioning to the next state, s', assuming action a is taken in state s
  V(s') is the value of a future state, s'
  B is the discount factor

This is complicated: B * Sum( P(s',r | s, a) * V(s') )

To make it easier, I want to use a Monte Carlo method of solving this problem. A Monte Carlo method will run the ADP optimization algorithm numerous (hundreds?) of times. According to the Law of Large Numbers, an expected value will approach its mean value as the number of iterations increases. Theoretically, the expected value = mean value if the problem is solved an infinite number of times.

When solving the problem as a Monte Carlo simulation, the Value Function Approximation simplifies to:

V(s,t) = min( Electricity Cost(s,t,a) + Average Future Value)

And the goal of the algorithm is to find:

Optimal Policy(s,t) = argmin( V(s,t) )




