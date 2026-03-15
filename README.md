MULTI-ROBOT PLANNING AND COORDINATION UNDER UNCERTAINTY

A simulation framework for studying multi-UAV/robot coordination under stochastic conditions and resource constraints. Directly aligned with research problems in AI-enabled autonomous systems and national security robotics (Mission Sudarshan Chakra, Atmanirbhar Bharat).

PROBLEM

Coordinated teams of mobile robots operating in uncertain environments face three compounding challenges: stochastic action failures (hostile interference, technical glitches), unpredictable battery drain, and intermittent communication loss. Naive single-robot policies break down quickly under these conditions. This project studies how different planning strategies perform as uncertainty increases.

ENVIRONMENT

A 2D grid workspace with obstacles, charging stations, and task locations. Each robot has a limited battery that depletes with movement. Three uncertainty events are modeled:

- Action failure- robot fails to execute its intended move (probability 0.10)
- Battery spike - unexpected extra drain due to interference (probability 0.05)  
- Communication loss - robot loses coordination signal for one timestep (probability 0.08)

AGENT POLICIES

Random -baseline agent, picks actions uniformly at random.

Greedy - navigates to the nearest pending task via BFS. Diverts to the nearest charging station when battery falls below a safety threshold. No inter-robot coordination.

Coordinated -multi-robot task allocation with battery-aware planning. Robots bid on tasks based on distance and projected battery cost. A safety buffer ensures robots can always reach a charger after completing a task. Replans periodically to adapt to uncertainty events.

RESULTS

Coordinated agents consistently outperform greedy and random agents on task completion rate, especially as uncertainty increases. The battery trajectory plots show coordinated agents maintaining safe battery levels throughout, while random agents frequently die mid-episode.

SETUp


pip install numpy matplotlib
cd src
python main.py

Results saved to `results/` automatically.

PROJECT STRUCTURE

The project has four core source files inside src/. environment.py handles the grid, robots, battery tracking, and all uncertainty events. agents.py contains the three planning policies. simulation.py runs episodes and generates plots. main.py ties everything together and runs all experiments. Results are saved automatically to the results/ folder.

REFERENCES

Kundu, T. et al. (2023). Multi-robot Planning under Uncertainty. IROS 2023.

Cao, Y. et al. (1997). Cooperative Mobile Robotics: Antecedents and Directions. Autonomous Robots, 4(1), 7–27.

Oliehoek, F. & Amato, C. (2016). A Concise Introduction to Decentralized POMDPs. Springer.