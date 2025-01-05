# Automated Hex Nut Fastening using Reinforcement Learning

## Overview
This project implements a reinforcement learning (RL) approach to automate the fastening of hex nuts onto threaded studs using an articulated robot arm. The goal was to use a Deep Q-Network (DQN) algorithm to improve upon the classical controls approach and create a system capable of adapting to environmental changes without extensive manual tuning.

## Contents
Due to Controlled Unclassified Information (CUI) restrictions, not all parts of the created code are included in this repository. However, the following components are available:
- **Iteration Data**: Contains training iterations, corresponding environment setup, and model checkpoints.
- **Documentation**: Includes the project report and related figures for reference.

 ## Directory Structure
```plaintext
Automated-Hex-Nut-Fastening/
├── InitialTesting/
│   ├── experience_data.csv
│   ├── policy_net.pth
│   ├── rl_environment.py
│   ├── target_net.pth
│   ├── training_metrics_rewards.csv
│   ├── training_metrics_state1.csv
│   └── training_metrics_state2.csv
├── Iteration1/
│   ├── experience_data.csv
│   ├── policy_net.pth
│   ├── rl_environment.py
│   ├── target_net.pth
│   ├── training_metrics_rewards.csv
│   ├── training_metrics_state1.csv
│   └── training_metrics_state2.csv
├── Iteration2/
│   └── (similar structure as above)
├── Iteration3/
│   └── (similar structure as above)
├── Iteration4/
│   └── (similar structure as above)
├── FinalIteration/
│   └── (similar structure as above)
├── docs/
│   ├── report.pdf
│   └── figures/
│       ├── figure1.png
│       ├── figure2.png
│       └── ...
├── README.md
└── requirements.txt
```

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Automated-Hex-Nut-Fastening.git
    ```
2. Navigate to the project directory:
   ```bash
   cd Automated-Hex-Nut-Fastening
    ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
    ```

## Usage
1. Training Iterations:
* Navigate to the desired iteration folder (e.g., FinalIteration/).
* Use rl_environment.py to set up the environment.
* Load the saved policy and target network (policy_net.pth and target_net.pth) to test or retrain the model.

2. Metrics:
* Analyze training progress using the CSV files (training_metrics_rewards.csv, training_metrics_state1.csv, training_metrics_state2.csv) in each iteration folder.
  
## Report and Figures

The full project report is available in the docs/ directory: report.pdf. Figures from the report, including architecture diagrams and performance metrics, can be found in docs/figures/.

## Known Limitations

* Not all code is included due to CUI restrictions.
* Training on the physical system requires significant time investment.
* Further tuning of the reward scheme may enhance performance.

## References

Refer to the project report for a complete list of references and acknowledgments.
