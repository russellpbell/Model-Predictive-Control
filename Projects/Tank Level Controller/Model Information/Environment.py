import numpy as np
import pandas as pd
import torch

class environment():
    """
    -----------------------------------------------------------------------------------------------------------------------------
    Title:          Tank Environment
    Author:         Russell Bell
    Date:           August 10, 2021
    -----------------------------------------------------------------------------------------------------------------------------
        Description:
            The environment describes a single tank of uniform cross-sectional area. The volume in the tank is decribed by the
            following equations:
                                    A_T * h'(t) = Q_in(t) - Q_out(t)
            The fill rate Q_in(t) is described by taking the pump voltage, u(t), multiplied by a constant gain, K_v. The outlet
            flow is described by the following equation derived from the Bernouli equation:
                                    Q_out(t) = A_out * sqrt(2 * g * h(t))
            Thus, we derive the following equation to describe the system:
                                    A_T * h'(t) = Q_max * A_in - A_out * sqrt(2 * g * h(t))
            Though the first principle models are available for this system, a neural network is used to approximate these functions
            to prove out the concept of using neural networks for controlling environments.

            Reference: https://www.scirp.org/journal/paperinformation.aspx?paperid=102677
        
        Observation:
            Observation         Min             Max         Max Move        Type
            Tank Level          0               20          NaN             DV
            Inlet Valve Pos     0               100         10              MV
            Outlet Valve Pos    0               100         NaN             DV
            Velocity            -Inf            Inf         NaN             CV
        
        Actions:
            The action space is discretized into intervals determined by the "move_intervals" variable,
            which is user defined.

        Reward:
            - Reward of (+) is awarded if the move results in being closer from the goal value
            - Reward of (++) is awarded if the move causes the goal value to be reached
            - Reward of (-) is awarded if the move results in being further away from the goal value
            - Reward of (--) is awarded if the move results in a violation of any constraint
        -----------------------------------------------------------------------------------------------------------------------------
    """

    project_title = 'Tank Level Controller'

    def __init__(self):
        
        self.independent_variables = pd.DataFrame({
            'Variable':                 np.array(['level',      'A_in',         'A_out']), 
            'UOM':                      np.array(['feet',       'percent',      'percent']),
            'Model Group':              np.array([1,            1,              1]),
            'Max Move':                 np.array([np.nan,       10,             np.nan]),
            'High Control Limit':       np.array([np.nan,       np.nan,         np.nan]),
            'Low Control Limit':        np.array([np.nan,       np.nan,         np.nan]),
            'High Data Limit':          np.array([20,           100,            100]),
            'Low Data Limit':           np.array([0,            0,              0]),
            'Settling Time (periods)':  np.array([0,            0,              0]),
            'Dead Time (periods)':      np.array([0,            0,              0])
            })

        self.dependent_variables = pd.DataFrame({
            'Variable':                 np.array(['velocity']), 
            'UOM':                      np.array(['feet/step']),
            'Model Group':              np.array([1]),
            'Max Move':                 np.array([np.nan]),
            'High Control Limit':       np.array([np.nan]),
            'Low Control Limit':        np.array([np.nan]),
            'High Data Limit':          np.array([np.nan]),
            'Low Data Limit':           np.array([np.nan])
            })
        
        self.goals = pd.DataFrame({
            'Variable':                 np.array(['level']), 
            'Target Value':             np.array([4])
            }).set_index('Variable') # Defining control goals
        
        self.response_speed = np.zeros(len(self.dependent_variables))

        self.observation_space_df = pd.DataFrame({'Observation': np.append(self.independent_variables['Variable'].to_numpy(), self.dependent_variables['Variable'].to_numpy()), 
            'Min': np.append(self.independent_variables['Low Data Limit'].to_numpy(), self.dependent_variables['Low Data Limit'].to_numpy()),
            'Max': np.append(self.independent_variables['High Data Limit'].to_numpy(), self.dependent_variables['High Data Limit'].to_numpy()),
            'Max Move': np.append(self.independent_variables['Max Move'].to_numpy(), self.dependent_variables['Max Move'].to_numpy())}).set_index('Observation')
        
        self.mv_indices = self.independent_variables[self.independent_variables['Max Move'] > 0].index.to_numpy()

    def step(self, action, move_intervals, raw_state, prediction_models, training_avgs, training_stds):
        def normalize(raw_value, avg, std_dev):
            return ((raw_value - avg) / std_dev)
        def denormalize(norm_value, avg, std_dev):
            return ((norm_value * std_dev) + avg)
        def move_check(prev_value, new_value, max_value, min_value, max_move):
            if abs(new_value - prev_value) > max_move:
                new_value = prev_value + (max_move * (1 if (new_value - prev_value) > 0 else -1))
            
            new_value = round(min(max(new_value, min_value), max_value), 2)

            return new_value      
        def move_reward(value, response_speed, goal):
            
            if abs((value + response_speed) - goal) < 0.01:
                # Value is at goal
                reward = 2
            elif (response_speed/ (value - goal)) > 0:
                # Value is moving away from its goal
                reward = -1 * abs(response_speed)
            elif (response_speed / (value - goal)) < 0:
                # Value is moving toward its goal
                if (response_speed / ((value + response_speed) - goal)) < 0:
                    # Move will not overshoot the goal
                    reward = 1 * abs(response_speed)
                else:
                    # Move will overshoot the goal
                    reward = -1 * abs(response_speed)
            else:
                # Value is not at goal and there's no movement
                reward = -1
            
            return reward

        # Defining the prediction model for the environment along with the averages and standard deviations
        # of the training set, which will be used to normalize inputs.
        response_speed_model = prediction_models[0]
        response_speed_model_avgs = training_avgs[0]
        response_speed_model_stds = training_stds[0]

        # Initialize the env_state variable to use in the remaining step calculations
        env_state = np.zeros(self.state.shape)
        prev_state = np.zeros(self.state.shape)
        for i in range(len(env_state)):
            env_state[i] = prev_state[i] = self.state[i]

        env_state[0] = float(env_state[0]) + float(self.response_speed[0]) # Case-study specific

        # Calculate the action to take for each MV and check to ensure the action respects user-defined constraints
        for i in range(len(self.mv_indices)):
            env_state[self.mv_indices[i]] = action[i] * ((self.observation_space_df.iloc[self.mv_indices[i], 1] - self.observation_space_df.iloc[self.mv_indices[i], 0]) / (move_intervals - 1))

            env_state[self.mv_indices[i]] = move_check(raw_state[self.mv_indices[i]], env_state[self.mv_indices[i]], 
                self.observation_space_df.iloc[self.mv_indices[i], 1], self.observation_space_df.iloc[self.mv_indices[i], 0], 
                    self.observation_space_df.iloc[self.mv_indices[i], 2])

        # Normalizing inputs
        norm_state = np.zeros(self.state.shape)
        for i in range(len(norm_state)):
            norm_state[i] = normalize(env_state[i], response_speed_model_avgs.iloc[0, i], response_speed_model_stds.iloc[0, i])
        
        # Calculating response speed
        x = torch.from_numpy(norm_state).to(torch.float32)
        self.response_speed[0] = denormalize(float(response_speed_model(x)), response_speed_model_avgs.loc[0, 'velocity'], response_speed_model_stds.loc[0, 'velocity']) # Case-study specific

        # Calculate the reward for the controller move with respect to the goal
        reward = move_reward(env_state[0], self.response_speed[0], self.goals.loc['level', 'Target Value']) # Case-study specific

        # Add penalties for violating constraints
        exceedance_count = 0
        for i in range(len(env_state)):
            exceedance_count += (1 if (env_state[i] > self.observation_space_df.iloc[i, 1]) else 0)
            exceedance_count += (1 if (env_state[i] < self.observation_space_df.iloc[i, 0]) else 0)
        reward += (-1 * exceedance_count)

        return env_state, reward