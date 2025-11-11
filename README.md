# Tank Level Controller

This repository implements a Model Predictive Controller (MPC) using neural networks to manage tank level in a simulated environment. It approximates first-principle models with NNs for process control, suitable for chemical engineering applications like petrochemicals.

## Features
- **Environment Simulation**: Models a single tank with inlet/outlet flows based on Bernoulli's equation.
- **Neural Network Prediction**: Trains models to predict dependent variables (e.g., velocity) from independents (e.g., level, valve positions).
- **Controller Logic**: Optimizes actions over prediction horizons, with rewards for goal proximity and penalties for constraint violations.
- **GUI Interface**: Tkinter-based UI for training models and running control scenarios.
- **Visualizations**: Generates plots for training data fits, SHAP explanations, and control predictions.

## Requirements
- Python 3.12+
- Libraries: `numpy`, `pandas`, `torch`, `sklearn`, `shap`, `matplotlib`, `tqdm`, `tkinter`

Install via:
```
pip install numpy pandas torch scikit-learn shap matplotlib tqdm
```

## Usage
1. **Setup**: Place data files in `/Projects/<ProjectName>/Data/` (e.g., `hist_training.csv`, `initial_data.csv`).
2. **Run GUI**: Execute `MPC_GUI.py` to select project and mode (Model Trainer or Controller).
   - Trainer: Specify epochs to build NN models.
   - Controller: Set period size, prediction periods, moves per step, and starting timestamp.
3. **Outputs**: Models saved in `/Models/`, plots in `/Plots/`, predictions in `/Data/pred_data.csv`.

## Project Structure
- `Environment.py`: Defines tank dynamics, observations, actions, and rewards.
- `MPC.py`: Handles data extraction, model training, visualization, and control logic.
- `MPC_GUI.py`: Launches GUI for training and control execution.

## Example
Train a model:
- Select "Model Trainer" in GUI.
- Enter epochs (e.g., 1000).
- Models and visuals generated automatically.

Run controller:
- Select "Controller".
- Input parameters (e.g., period size=1, periods=100, moves=10, start=0).
- View predicted control actions and plots.

## Limitations
- Discretized action space; no continuous optimization.
- Single-model group; extend for multi-variable systems.
- No real-time integration; simulation-only.

## References
- Deep RL for Process Control: [arXiv:2004.05490](https://arxiv.org/pdf/2004.05490.pdf)
- Non-Linear Tank Control: [SCIRP Paper](https://www.scirp.org/journal/paperinformation.aspx?paperid=102677)
- Model-Based NN Control: [Medium Article](https://medium.com/swlh/model-based-rl-for-nonlinear-dynamics-control-a-case-study-70c31810f255)

## License
MIT License. See [LICENSE](LICENSE) for details.
