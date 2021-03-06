The project is about using reinforcement learning techniques to find an optimal transmission policy for harvesting powered sensors. We consider a controlled dynamical system where a wireless sensor transmits its measurements to a controller over a communication channel. We assume that the sensor has a harvesting element to extract energy from the environment and store it in a rechargeable battery for future use. If the sensor transmits its measurements (it uses some energy for this), then the controller obtains recent data which improves the system performance. If at a certain time slot, the sensor decides not to transmit, it saves more energy, but the controller does not update its information and worsens the performance. The goal is to find a suitable transmission policy providing both an acceptable quality of control and efficient energy consumption. The project will contain three main modules:

1) Represent the overall model (dynamical system model, energy harvesting model, communication channel gain model, battery model) as a Markov decision process (MDP) with possible actions {0 - not transmit, 1 -transmit} and define transition probability matrix (N-dimensional).

2) Realize an iterative algorithm (value-iteration) solving the Bellman equations to obtain an optimal policy.

3) Try the obtained policy by simulating the closed-loop system. Plot the results.  
