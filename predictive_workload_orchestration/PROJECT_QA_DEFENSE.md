# Project Defense Q&A: Predictive Workload Orchestration (CloudVerse)

This document provides a detailed, Question and Answer (Q&A) style breakdown of the entire project. It covers the technology stack, architectural decisions, machine learning implementation, and experimental results. It is designed to help explain "what", "why", and "how" the project works in comprehensive detail.

---

## Part 1: Technology Stack & Architecture

**Q1: What programming languages and core frameworks did you use for the frontend and backend, and why?**  
**Answer:**  
*   **Backend:** Python (FastAPI). Python is the undeniable industry standard for Data Science and Machine Learning. Since the core of this project relies on PyTorch (for the Deep Q-Network) and Prophet (for time-series forecasting), Python was mandatory. FastAPI was chosen because of its excellent support for asynchronous operations and WebSockets, which are critical for streaming real-time simulation data.
*   **Frontend:** TypeScript and React (bundled with Vite). TypeScript provides strict type safety, which is crucial when handling complex, rapidly updating JSON payloads from the backend simulation. React's component-based architecture is perfect for building the complex UI panels.
*   **3D Visualization:** Three.js and React Three Fiber (R3F). Standard HTML/CSS cannot handle rendering thousands of moving objects (like cars representing network traffic) without severe performance lag. R3F offloads this rendering to the GPU via WebGL, allowing us to build a high-performance "Digital Twin" of the datacenter.

**Q2: How do the frontend and backend communicate?**  
**Answer:**  
They communicate primarily through **WebSockets**. The Python backend runs a continuous simulation loop (the `SimulationBridge`). At every tick of the simulation, it calculates the server loads, executes the AI scaling actions, and packages the entire state into a JSON event payload. It broadcasts this payload over the WebSocket. The React frontend listens to this WebSocket, catches the data using the `Zustand` state management library, and immediately updates the 3D graphics and 2D charts in real-time without requiring a page refresh. For one-off commands (like injecting a "Flash Sale" or "Host Failure"), the frontend uses standard REST API `POST` requests.

---

## Part 2: Machine Learning & Research Pipeline

**Q3: Explain the role of Meta's Prophet in this project.**  
**Answer:**  
Prophet is a time-series forecasting tool. In cloud computing, reacting to a traffic spike *after* it happens leads to dropped requests and SLA (Service Level Agreement) violations because booting up new Virtual Machines takes time. We use Prophet to analyze historical CPU utilization (from the Alibaba Cluster trace dataset) and predict *future* demand. This gives our orchestration system a "lookahead" signal, allowing it to proactively scale resources before the spike actually hits.

![Prophet Forecast Dashboard](results/visualizations/forecast_dashboard.png)

**Q4: How does the Deep Q-Network (DQN) agent work in your system?**  
**Answer:**  
The DQN is a Reinforcement Learning agent built using PyTorch and Gymnasium. 
Instead of using static, hardcoded rules (like "If CPU > 80%, add 1 VM"), the DQN *learns* the optimal scaling policy through trial and error in a simulated environment.
*   **State:** The agent observes the current CPU utilization, queue length, and the Prophet forecast.
*   **Action:** It decides whether to scale up, scale down, or hold the current number of VMs.
*   **Reward:** It receives a positive reward for maintaining target CPU utilization (cost efficiency) and a heavy negative penalty for violating SLAs (dropped requests) or causing excessive scaling churn.
Over 200 training episodes, the neural network learns which actions maximize long-term rewards across different workload scenarios.

![DQN Training Dashboard](results/visualizations/training_dashboard.png)

**Q5: What is HARS (Hybrid Adaptive Reward Scheduling) and why was it necessary?**  
**Answer:**  
Standard reinforcement learning uses fixed reward weights. However, we found that if the workload suddenly becomes very difficult (e.g., a massive unpredictable spike), the agent might struggle to balance cost vs. performance. 
HARS is a custom algorithmic component we designed that dynamically adjusts the reward function on the fly. If the agent is experiencing high SLA violations (a "hard" phase), HARS increases the penalty weight for SLA breaches, forcing the agent to prioritize performance. Once the system stabilizes, HARS relaxes the SLA weight to encourage the agent to focus on cost efficiency. This resulted in a much more stable and robust policy.

![HARS Dashboard](results/visualizations/hars_dashboard.png)

---

## Part 3: Implementation & Hardware Details

**Q6: What hardware and software environment was used to train the models?**  
**Answer:**  
*   **Hardware:** An x86_64 CPU and an NVIDIA GeForce RTX 5090 GPU. The GPU significantly accelerated the tensor operations in PyTorch during the DQN training phase.
*   **Software Versions:** Python 3.12.3, PyTorch 2.11.0, Gymnasium 0.29.1, and Prophet 1.1.5. The final digital twin infrastructure validation was handled by CloudSim Plus running on Java 17+.

**Q7: What were the key hyperparameters used for training the DQN?**  
**Answer:**  
Through grid-search tuning, we settled on:
*   **Learning Rate:** 0.001 (using the Adam optimizer)
*   **Batch Size:** 64
*   **Replay Buffer Size:** 50,000 transitions
*   **Target-Network Update Interval:** 500 steps
*   **Epsilon (ε) Schedule:** Linear decay from 1.0 (pure exploration) down to 0.05 over the first 10,000 steps.
*   **Discount Factor (γ):** 0.99 (encouraging long-term planning).

---

## Part 4: Experimental Results & Analysis

**Q8: Did the time-series forecasting (Prophet) perform accurately?**  
**Answer:**  
Yes. Evaluated on a hold-out test set at a 5-minute resolution, Prophet achieved a Mean Absolute Percentage Error (MAPE) of 20.65%. In the context of highly variable and noisy cloud workloads, this is considered a strong result. It successfully captured both daily business-hour cyclic patterns and weekly seasonality, providing a reliable predictive signal to the DQN.

**Q9: How did the full system (Prophet + DQN + HARS) compare to standard rule-based auto-scaling?**  
**Answer:**  
The results showed a massive improvement in system reliability.
*   **Static Provisioning** (doing nothing) resulted in an SLA violation rate of ~33%.
*   **Threshold-based rules** aggressively tried to cut costs but suffered from severe oscillatory behavior (scaling up and down too late), resulting in a catastrophic 99.65% SLA violation rate.
*   **Our Full System (Prophet + DQN + HARS)** reduced the SLA violation rate to just **3.40%**. 

While our system had a marginally higher estimated cost ($6.51 vs $6.00 for static) due to proactive provisioning, it virtually eliminated dropped requests. A Welch's t-test confirmed that our RL system's outperformance over the baselines was statistically significant (p = 0.0064). 

![Ablation Results](results/ablation/plots/ablation_bars.png)
![SLA vs Cost Tradeoff](results/ablation/plots/sla_cost_tradeoff.png)

**Q10: What is the final conclusion of the project?**  
**Answer:**  
The project successfully proves that combining time-series forecasting (Prophet) with an adaptive Reinforcement Learning agent (DQN + HARS) creates a vastly superior cloud orchestration policy. By anticipating traffic rather than just reacting to it, the CloudVerse system reduces SLA violations by nearly 90% compared to traditional static baselines, while dynamically visualizing these complex datacenter operations in a real-time 3D digital twin.
