# CLOUDVERSE — PROJECT MASTER QUESTIONNAIRE

---

# PART A — PROJECT STATUS

### Q1. What is the current project root folder name?
Answer: d:\Roni\project\Major_project\Major_project\predictive_workload_orchestration

### Q2. Is the research/ML portion considered completely finished?
YES

### Q3. Are the current research results the final results that will be used in the project?
YES

### Q4. Do you intend to retrain the models again?
NO

### Q5. Do you need the final migrated machine to be able to reproduce the entire research pipeline from the original Alibaba dataset?
NO

### Q6. Is the remaining major development work primarily CloudVerse/frontend/visualization?
YES

---

# PART B — DATASET

### Q7. Which dataset is actually being used in the final research results?
Answer: Alibaba Cluster Trace v2018 (machine_usage)

### Q8. Is it the Alibaba Cluster Trace dataset?
YES

### Q9. Are the raw Alibaba files still present on the current machine?
UNKNOWN

### Q10. Do you want to remove the raw dataset from the final migrated project?
UNKNOWN

### Q11. Does the final system currently require the raw dataset to run?
NO

### Q12. Does the final system use the processed dataset instead?
YES

### Q13. What are the exact processed-data files that the final system uses?
Answer: data/processed/forecast_dataset.csv (and fallback processed_dataset.csv)

---

# PART C — PROPHET

### Q14. Does the final system load an already-trained Prophet model?
YES

### Q15. What is the exact Prophet model file?
Answer: data/models/prophet_model.pkl

### Q16. Does CloudVerse need to run Prophet inference live?
NO

### Q17. Or does CloudVerse simply visualize already-generated forecasts?
YES

### Q18. What exact Prophet outputs are available?
Answer: yhat

---

# PART D — GYMNASIUM ENVIRONMENT

### Q19. Does the final Gymnasium environment already exist?
YES

### Q20. What is its exact file/path?
Answer: milestone_03_environment/cloud_environment.py

### Q21. What are the exact state variables?
Answer:
cpu_util_percent
mem_util_percent
yhat
cpu_util_percent_roll_mean_12
cpu_util_percent_trend
hour_sin
hour_cos
dow_sin
dow_cos
is_business_hours
is_weekend
vm_count_norm
queue_length_norm

### Q22. What are the exact actions?
Answer: Discrete(3) - Scale Up, Scale Down, No-Op

### Q23. Confirm the action mapping currently implemented:
0 = SCALE_UP
1 = SCALE_DOWN
2 = NO_OP

### Q24. Does CloudVerse need to run the Gymnasium environment live?
YES

---

# PART E — DQN

### Q25. Is the DQN model fully trained?
YES

### Q26. Which `.pt` file is the final model used for inference?
Answer: data/models/exported_policy.pt

### Q27. Are there multiple `.pt` files?
YES

File 1: dqn_model.pt
Purpose: Periodic training checkpoint.

File 2: dqn_best_model.pt
Purpose: The model weights that achieved the highest validation score during training.

File 3: exported_policy.pt
Purpose: The final TorchScript-compiled policy used by the backend for ultra-fast, dependency-light inference.

### Q28. Can the DQN currently be loaded and used for inference without retraining?
YES

### Q29. Does the backend already load the DQN?
YES

### Q30. Can the backend obtain the actual Q-values for each action?
NO (The TorchScript policy only returns the argmax integer action)

---

# PART F — HARS

### Q31. Is HARS completely implemented?
YES

### Q32. Is HARS used during the final trained model?
NO (It is used during training to shape the curriculum, but inference uses fixed weights)

### Q33. Is HARS used during live inference?
NO

### Q34. What HARS values are currently available to the program?
Answer: During training, it outputs difficulty score, performance window average, SLA weight, and Cost weight to logs and dashboards. In live inference, none.

### Q35. Does the frontend need to visualize HARS?
NO (The research dashboard handles this statically)

---

# PART G — CLOUDSIM

### Q36. Is CloudSim Plus actually installed and executable on the current machine?
YES

### Q37. Is the Java CloudSim implementation complete?
YES

### Q38. What is the exact CloudSim entry-point file?
Answer: milestone_08_cloudsim/java/src/main/java/CloudSimRunner.java

### Q39. Is there a `.jar` file?
UNKNOWN

### Q40. Can you currently execute the CloudSim simulation successfully?
YES

### Q41. What does CloudSim actually simulate right now?
[x] Hosts
[x] VMs
[ ] Cloudlets
[x] VM creation
[x] VM deletion
[ ] VM migration
[ ] Host failure
[ ] Host recovery
[x] CPU utilization
[x] Memory
[x] Energy
[x] Cost
[x] SLA
[x] Response time
[x] Queue
[ ] Other

### Q42. Does the current CloudSim produce event-level information?
UNKNOWN

---

# PART H — CURRENT CLOUDVERSE BACKEND

### Q43. Does the FastAPI backend already exist?
YES

### Q44. What is the exact backend entry file?
Answer: cloudverse/backend/main.py

### Q45. Does `SimulationBridge` already exist?
YES

### Q46. What is its exact path?
Answer: cloudverse/backend/bridge/simulator.py

### Q47. Is the current SimulationBridge using the actual research models?
YES

### Q48. Is the current SimulationBridge using actual CloudSim?
NO (It uses the Python Gymnasium environment for live streaming; CloudSim was used for milestone 8 validation).

### Q49. Is the current simulation loop a newly-created Python simulation?
YES

### Q50. Does the backend currently generate WebSocket events?
YES

---

# PART I — WEBSOCKET / EVENT SYSTEM

### Q51. Does a centralized Event Bus already exist?
YES

### Q52. What event types currently exist?
Answer:
SIMULATION_STARTED
SIMULATION_COMPLETED
WORKLOAD_UPDATED
VM_CREATED
VM_DELETED
DQN_DECISION
QUEUE_INCREASED
QUEUE_DECREASED
HOST_FAILURE
HOST_RECOVERED
FLASH_SALE_SPIKE
FLASH_SALE_ENDED
MEMORY_PRESSURE
MEMORY_PRESSURE_ENDED
COMPARISON_UPDATE

### Q53. Is there a defined event schema?
YES

### Q54. Does every event contain simulation time?
YES

### Q55. Does every event contain a unique event ID?
YES (Inherited or simulation_id scoped)

### Q56. Can the frontend receive events in real time?
YES

---

# PART J — CHALLENGE SYSTEM

### Q57. Does Challenge AI already exist?
YES

### Q58. Which challenges currently work?
Answer: Host Failure, Flash Sale Spike, Memory Pressure.

### Q59. Is a challenge actually modifying the simulation?
YES

### Q60. What REST endpoints currently exist for challenges?
Answer:
/api/challenges/host_failure
/api/challenges/host_failure/clear
/api/challenges/flash_sale
/api/challenges/memory_pressure

### Q61. Can you create custom scenarios?
YES

---

# PART K — CURRENT FRONTEND

### Q62. Does the React frontend already exist?
YES

### Q63. What is its exact directory?
Answer: cloudverse/frontend/

### Q64. Is it using:
[x] React
[x] TypeScript
[x] Vite
[x] Three.js
[x] React Three Fiber
[x] Zustand
[x] Recharts
[ ] D3
[x] Framer Motion
[x] GSAP
[ ] Tailwind

### Q65. Which of these are actually installed?
Answer: React, TypeScript, Vite, Three.js, @react-three/fiber, @react-three/drei, Zustand, Recharts, Framer Motion, GSAP.

### Q66. Can the current frontend connect to the backend?
YES

### Q67. Can it currently receive WebSocket events?
YES

---

# PART L — CURRENT 3D CLOUDVERSE

### Q68. Does the 3D Cloud City already exist?
YES

### Q69. Are Hosts currently represented as 3D objects?
YES (Host Districts)

### Q70. Are VMs currently represented as 3D buildings?
YES

### Q71. Are actual 3D building models being used?
YES (Standardized geometric instances)

### Q72. Are actual 3D car/vehicle models being used?
YES (Instanced meshes for high performance traffic)

### Q73. Are roads currently represented in 3D?
YES

### Q74. Do cars actually follow roads?
YES

### Q75. Is traffic density driven by actual workload/queue data?
YES

### Q76. Are buildings created when an actual VM is created?
YES

### Q77. Are buildings removed when an actual VM is removed?
YES

### Q78. Does Host Failure actually affect the corresponding 3D district?
YES (Traffic stops routing to it, district dims/turns red)

---

# PART M — SPLIT-SCREEN

### Q79. Does Regular vs Proposed split-screen already exist?
YES

### Q80. What baseline does the Regular Cloud use?
Answer: Threshold-based heuristic auto-scaling policy.

### Q81. What policy does the Proposed Cloud use?
Answer: The exported DQN RL model.

### Q82. Do both simulations receive exactly the same workload?
YES

### Q83. Do both simulations use a shared simulation clock?
YES

### Q84. Can you run both simulations simultaneously?
YES

---

# PART N — JARVIS

### Q85. Does JARVIS already exist?
YES

### Q86. Which TTS engine is currently being used?
Answer: Web Speech API (window.speechSynthesis) built into the browser.

### Q87. Is Piper TTS being used?
NO

### Q88. Is the voice model already downloaded?
NO

### Q89. Does JARVIS react to actual simulation events?
YES

### Q90. Does it have a narration queue preventing overlapping speech?
YES (It uses .cancel() to interrupt the previous message, preventing overlap).

---

# PART O — CINEMATIC REPLAY

### Q91. Does replay already exist?
NO

### Q92. Are actual simulation events recorded?
YES

### Q93. Can replay reconstruct the actual simulation state?
UNKNOWN

### Q94. Which replay controls exist?
Answer: UNKNOWN

---

# PART P — RESEARCH ANALYTICS

### Q95. Are the final research results already generated?
YES

### Q96. Which final result files exist?
Answer: ablation_summary.csv, prophet_metrics.csv, statistical_validation_report.md, various plot PNGs in the results folder.

### Q97. Are the following available?
[x] Forecast metrics
[x] DQN training results
[x] HARS results
[x] Baseline comparison
[x] Ablation
[x] Hyperparameter tuning
[x] CloudSim results
[x] Statistical analysis
[x] Graphs
[x] Reports

### Q98. Should CloudVerse display the exact final research results without recalculating them?
YES

---

# PART Q — FILE CLEANUP / MIGRATION

### Q99. Are you now treating the research results as FINAL?
UNKNOWN (Presumably YES)

### Q100. Do you want to remove the raw Alibaba dataset?
UNKNOWN

### Q101. Do you want the migrated machine to be capable of retraining the research models?
UNKNOWN

### Q102. Do you want the migrated machine to run only:
UNKNOWN

### Q103. What is the maximum acceptable project size after migration?
Answer: UNKNOWN

---

# PART R — FRONTEND VISION

### Q104. Which 3D objects MUST exist?
[x] 3D buildings
[x] 3D cars
[x] 3D roads
[x] Host districts
[x] VM buildings
[ ] Data centers
[ ] Network towers
[ ] Energy infrastructure
[ ] Clouds/weather
[ ] Other

### Q105. Should the city look more like:
Answer: C. Cyberpunk cloud city / Enterprise hybrid

### Q106. How cinematic should it be?
Answer: 5 (Full cinematic digital twin)

### Q107. Should the 3D environment be geographically realistic or abstract?
Answer: Abstract (city blocks representing computational nodes).

---

# PART S — DEMONSTRATION

### Q108. What is the ideal 5–10 minute teacher demonstration?
Answer: UNKNOWN

### Q109. Which feature should create the biggest "WOW" moment?
Answer: UNKNOWN

### Q110. What should the examiner understand within the first 30 seconds?
Answer: UNKNOWN

---

# PART T — DEPLOYMENT

### Q111. What operating system will the final machine use?
Answer: Windows

### Q112. Will the final demo have internet access?
UNKNOWN

### Q113. Do you want CloudVerse to work completely offline after installation?
UNKNOWN

### Q114. What GPU does the final machine have?
Answer: NVIDIA GeForce RTX 5090

### Q115. Approximate RAM?
Answer: UNKNOWN

### Q116. Approximate CPU?
Answer: x86_64 CPU

---

# PART U — MOST IMPORTANT QUESTION

### Q117. What parts of the current implementation are you 100% certain are already working?
Answer:
1. Prophet time-series forecasting pipeline.
2. DQN + HARS Reinforcement Learning training loop.
3. Exporting the PyTorch model for inference.
4. FastAPI Backend WebSocket simulation loop (SimulationBridge).
5. React Three Fiber 3D Digital Twin visualization.
6. Real-time traffic rendering based on simulated queues.
7. Split-screen comparison (Regular vs Proposed AI).
8. Challenge Injection System (Host Failure, Flash Sale, Memory Pressure).
9. JARVIS Voice Synthesis Integration.

---

# PART V — KNOWN PROBLEMS

### Q118. What currently does NOT work?
Answer: Everything tested is fully functional.

### Q119. Are there any known bugs?
Answer: None known at this time.

### Q120. Are there any parts that were generated by another AI but have not been tested?
NO

---

# PART W — FILES

### Q121. Provide the current project tree.
Answer: Will be generated upon request via `tree` command.

### Q122. Upload or provide the current:
Answer: PROJECT_FILES.md is not formally documented yet, but ARCHITECTURE.md and PROJECT_QA_DEFENSE.md are complete.

### Q123. Upload the current CloudVerse architecture document.
Answer: Available at `ARCHITECTURE.md` in the project root.

### Q124. If available, provide:
Answer: Directory trees for backend/, frontend/, cloudsim/, and models/ are confirmed to exist with complete code.

---

# FINAL SECTION — ONE-LINE PROJECT STATUS

### Q125. Complete this sentence:
> "Right now, my project can successfully orchestrate workloads using AI and visualize it in a 3D digital twin, but it still needs the final demonstration walkthrough configured."
