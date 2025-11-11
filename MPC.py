import pandas as pd
import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from tkinter import *
from tkinter.ttk import *
import threading

def data_extractor(data_file, model_num, independent_variables, dependent_variables):
    """
        --------------------------------------------------------------------------------------------------------
        Title:    Data Extractor 
        Author:   Russell Bell
        Date:     April 15, 2021
        --------------------------------------------------------------------------------------------------------
        This program pulls dependent data and independent data into separate dataframes from a CSV file.
        --------------------------------------------------------------------------------------------------------
    """
    
    # Data is read into a dataframe from the CSV file and split into sepeate dataframes for the independent and 
    # dependent variables
    data_df = pd.read_csv(data_file)
    data_df = data_df.set_index('timestamp')
    independent_df = data_df[list(independent_variables[independent_variables['Model Group'] == model_num+1].loc[:,'Variable'])]
    dependent_df = data_df[list(dependent_variables[dependent_variables['Model Group'] == model_num+1].loc[:,'Variable'])]
    data_df = dependent_df.join(independent_df, on='timestamp')

    # The following "for" loop iterates through the data_df dataframe and shifts the data according to the specified amount
    for var in independent_variables[independent_variables['Model Group'] == model_num+1].loc[:,'Variable']:
        independent_variables.set_index('Variable', inplace=True)
        window_size = 1 if independent_variables['Settling Time (periods)'][var] == 0 else independent_variables['Settling Time (periods)'][var]
        data_df[var] = data_df[var].shift(independent_variables['Dead Time (periods)'][var])
        data_df[var] = data_df[var].rolling(window_size, min_periods=1, win_type='exponential').mean(tau=-(window_size-1) / np.log(0.01), center=window_size, sym=False)
        independent_df[var] = independent_df[var].shift(independent_variables['Settling Time (periods)'][var])
        independent_variables.reset_index(inplace=True)

    data_df.dropna(how='any', inplace=True) # Drops any rows with NaN values
    independent_df.dropna(how='any', inplace=True) # Drops any rows with NaN values

    return data_df, independent_df, dependent_df

def prediction_model_trainer(independent_variables, model_num, data_df, independent_var_list, dependent_var_list, model_folder, EPOCHS):
    """
        --------------------------------------------------------------------------------------------------------
        Title:    Model Trainer 
        Author:   Russell Bell
        Date:     July 15, 2021
        --------------------------------------------------------------------------------------------------------
        This program utilizes a neural network to predict dependent variable(s) using data from related 
        independent variable(s).
        --------------------------------------------------------------------------------------------------------
    """
    
    training_avgs_df = pd.DataFrame(index=[0], columns=independent_variables["Variable"].rename(None))
    training_stds_df = pd.DataFrame(index=[0], columns=independent_variables["Variable"].rename(None))
    
    # Normalizes the data via the Z-Score method and converts the dataframe into a numpy array so that it 
    # can be used in the MLPRegressor
    for column in data_df.columns:
        training_avgs_df[column] = data_df[column].mean()
        training_stds_df[column] = data_df[column].std()
        data_df[column] = (data_df[column] - data_df[column].mean()) / data_df[column].std()
    data_df.reset_index(inplace=True)
    data_np = data_df.to_numpy()
    
    # Sets up the appropriate arrays for plotting and regression purposes.
    dependent_np = data_np[:, 1:(len(dependent_var_list)+1)] # Creates a matrix of only the dependent data.
    independent_np = data_np[:,-1*(len(independent_var_list)):] # Creates a matrix of all the independent data.
    X_train, X_test, Y_train, Y_test = train_test_split(independent_np, dependent_np, train_size=0.75, random_state=23)

    independent_np = torch.from_numpy(independent_np).to(torch.float32)
    X_train = torch.from_numpy(X_train).to(torch.float32)
    Y_train = torch.from_numpy(Y_train).to(torch.float32)
    X_test = torch.from_numpy(X_test).to(torch.float32)
    Y_test = torch.from_numpy(Y_test).to(torch.float32)

    # torch can only train on Variable, so convert them to Variable
    x, y = torch.autograd.Variable(X_train), torch.autograd.Variable(Y_train)

    # Defining the network class
    prediction_model = torch.nn.Sequential(
        torch.nn.Linear(int(len(independent_var_list)), 7), # Input layer
        torch.nn.LeakyReLU(),
        torch.nn.Linear(7, 7), # Layer 1
        torch.nn.LeakyReLU(),
        torch.nn.Linear(7, 7), # Layer 2
        torch.nn.LeakyReLU(),
        torch.nn.Linear(7, 7), # Layer 3
        torch.nn.LeakyReLU(),
        torch.nn.Linear(7, 7), # Layer 4
        torch.nn.LeakyReLU(),
        torch.nn.Linear(7, 7), # Layer 5
        torch.nn.LeakyReLU(),
        torch.nn.Linear(7, 7), # Layer 6
        torch.nn.LeakyReLU(),
        torch.nn.Linear(7, 7), # Layer 7
        torch.nn.LeakyReLU(),
        torch.nn.Linear(7, 7), # Layer 8
        torch.nn.LeakyReLU(),
        torch.nn.Linear(7, 7), # Layer 9
        torch.nn.LeakyReLU(),
        torch.nn.Linear(7, 7), # Layer 10
        torch.nn.LeakyReLU(),
        torch.nn.Linear(7, int(len(dependent_var_list))), # Output Layer
    )

    optimizer = torch.optim.Adam(prediction_model.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    
    class Progress_screen(threading.Thread):

        def __init__(self, period):
            threading.Thread.__init__(self)
            self.period = period
        def callback(self):
            self.root.quit()
        def run(self):
            self.root = Tk()
            self.root.protocol("WM_DELETE_WINDOW", self.callback)
            self.root.title('Model Trainer')
            label = Label(self.root, text=('Progress: ' + str(self.period) + '/' + str(EPOCHS) + ' epochs completed'))
            label.pack(pady=10)
            self.root.mainloop()

    progress_screen = Progress_screen(period=0)
    progress_screen.start()

    # train the network
    for epoch in tqdm(range(EPOCHS), 'Training Prediction Model ' + str(model_num + 1)):
        prediction = prediction_model(x) # Generate predictions based on current gradients

        loss = loss_func(prediction, y) # Calculate the loss for the given predictions

        optimizer.zero_grad() # Clear current gradients
        loss.backward() # Calculate new gradients
        optimizer.step() # Apply new gradients

        progress_screen.period = epoch

    # Saves the model and returns the necessary variables
    torch.save(prediction_model, model_folder + 'prediction_model_' + str(model_num + 1) + '.pt')
    return prediction_model, independent_np, X_train, X_test, Y_train, Y_test, training_avgs_df, training_stds_df

def model_training_script(historical_data_file, model_folder, modelling_visuals_folder, independent_variables, dependent_variables, pred_epochs, show_visuals):
    training_amount = len(pd.read_csv(historical_data_file))

    # Train the prediction models
    prediction_models = []
    training_avgs = []
    training_stds = []
    for model_num in range(dependent_variables['Model Group'].max()):
        data_df, independent_df, dependent_df = data_extractor(historical_data_file, model_num, independent_variables, dependent_variables)
        training_data_df = data_df.tail(training_amount).copy()
        training_ind_df = independent_df.tail(training_amount).copy()
        training_dep_df = dependent_df.tail(training_amount).copy()
        prediction_model, X, X_train, X_test, Y_train, Y_test, training_avgs_df, training_stds_df = prediction_model_trainer(independent_variables, model_num, training_data_df, training_ind_df.columns, training_dep_df.columns, model_folder, pred_epochs)
        training_avgs_df.to_csv((model_folder + 'Training Averages - Model_' + str(model_num + 1) + '.csv'), index=False, mode='w')
        training_stds_df.to_csv((model_folder + 'Training Standard Deviations - Model_' + str(model_num + 1) + '.csv'), index=False, mode='w')
        training_avgs.append(training_avgs_df)
        training_stds.append(training_stds_df)
        prediction_models.append(prediction_model)
        model_visualizer(prediction_models[model_num], X, X_train, X_test, Y_train, Y_test, training_ind_df, training_dep_df, 
            modelling_visuals_folder, show_visuals_bool=show_visuals)

def model_visualizer(prediction_model, X, X_train, X_test, Y_train, Y_test, independent_df, dependent_df, save_folder, show_visuals_bool):
    """
        --------------------------------------------------------------------------------------------------------
        Title:    Model Visualizer 
        Author:   Russell Bell
        Date:     April 15, 2021
        --------------------------------------------------------------------------------------------------------
        This program creates visuals showing the model design and fit between the training and test sets. The
        program starts by displaying the SHAP summary plot which shows the directionality and relative impact
        that each variable has on the model. Then the program displays the dependence plot for each variable
        showing a visual representation of the relationship each variable has on the output. This dependence 
        plot also colors the values to show whether the displayed variable has an interaction with other
        variables.
        --------------------------------------------------------------------------------------------------------
    """

    def f(x):
        x = torch.from_numpy(x).to(torch.float32)
        return prediction_model(x).detach().numpy()

    # Creates a dataframe out of the normalized independent data. This makes it easier to handle later on.
    dataset_df = pd.DataFrame(X[:,:], columns=independent_df.columns)

    n = ((4.42**2 * 0.5 * (1 - 0.5)) / 0.05**2) # 99.999% confidence of a 5% MoE
    sample_size = int(n / (1 + n/len(dataset_df)))
    sampled_dataset_df = dataset_df.sample(sample_size)

    dataset_np = dataset_df.sample(sample_size).to_numpy(dtype='float32')
    
    # The following lines of code generate the SHAP values used for visualizations around model explainability.
    explainer = shap.KernelExplainer(f, dataset_np, feature_names=list(independent_df))
    shap_values = explainer.shap_values(dataset_np)

    # The following code creates values that are used to compare the fit between the testing and training sets.
    # The "for" loop creates the plots for each output.
    train_line = np.array(range(len(dependent_df.columns)), dtype=object)
    test_line = np.array(range(len(dependent_df.columns)), dtype=object)

    for dep_var in range(len(dependent_df.columns)):
        # The summary plot shows a high level the relationships and importance each variable has to a model.
        shap.summary_plot(shap_values[dep_var], dataset_np, feature_names=list(independent_df), show=False)
        plt.savefig(save_folder + 'summary_shap_plot_' + dependent_df.columns[dep_var] + '.png')
        if show_visuals_bool == True:
            plt.show()
        else:
            plt.close()

        # For each variable, an individual plot is created showing a graphical relationship between the variable
        # and the output. Additionally, it is colored based on the variable it has the most interaction with.
        for column in dataset_df.columns:
            shap.dependence_plot(column, shap_values[dep_var], sampled_dataset_df, show=False)
            plt.savefig(save_folder + 'shap_scatterplot_' + dependent_df.columns[dep_var] + '_' + column + '_' + '.png')
            if show_visuals_bool == True:
                plt.show()
            else:
                plt.close()

        train_line[dep_var] = np.array([min(Y_train[:,dep_var]), max(Y_train[:,dep_var])]) # Creates a line with a slope of 1, which represents a perfect fit
        test_line[dep_var] = np.array([min(Y_test[:,dep_var]), max(Y_test[:,dep_var])]) # Creates a line with a slope of 1, which represents a perfect fit
        Y_testPred = prediction_model(X_test).detach().numpy() # Create the predictions for the testing set
        Y_trainPred = prediction_model(X_train).detach().numpy() # Creates the predictions for the training set
        Y_testPred = Y_testPred.reshape(len(Y_test),len(dependent_df.columns))
        Y_trainPred = Y_trainPred.reshape(len(Y_train),len(dependent_df.columns))

        _, axis = plt.subplots(nrows=1,ncols=2,num=dependent_df.columns[dep_var] + ' Scatterplot') # Generates a figure with 2 charts

        # Creating the chart showing the fit of the training set
        axis[0].scatter(Y_train[:,dep_var], Y_trainPred[:,dep_var]) # Charting the predictions against actual values
        axis[0].plot(train_line[dep_var], train_line[dep_var], color='r') # Charting the line with a slope of 1, which represents a perfect fit
        axis[0].set_title('Training Data (R^2:  %.3f)'%r2_score(Y_train[:,dep_var], Y_trainPred[:,dep_var])) # Sets the title, which shows the r^2 for the training set
        
        # Creating the chart showing the fit of the testing set
        axis[1].scatter(Y_test[:,dep_var],Y_testPred[:,dep_var]) # Charting the predictions against actual values
        axis[1].plot(test_line[dep_var], test_line[dep_var], color='r') # Charting the line with a slope of 1, which represents a perfect fit
        axis[1].set_title('Testing Data (R^2:  %.3f)'%r2_score(Y_test[:,dep_var], Y_testPred[:,dep_var])) # Sets the title, which shows the r^2 for the training set
        
        plt.savefig(save_folder + 'train_test_scatterplots_' + dependent_df.columns[dep_var] + '.png')
        if show_visuals_bool == True:
            plt.show()
        else:
            plt.close()

def control_visualizer(data_df, independent_variables, independent_variables_uom, dependent_variables, dependent_variables_uom, visuals_file, environment):
    fig = plt.figure(figsize=(16,9), dpi=98, tight_layout=True)
    i = 0
    uoms = np.append(independent_variables_uom, dependent_variables_uom)
    for variable in np.append(independent_variables, dependent_variables):
        if variable in independent_variables:
            x, = np.where(independent_variables == variable)[0] + 1
        else:
            x, = max(len(independent_variables), len(dependent_variables)) + np.where(dependent_variables == variable)[0] + 1

        ax = fig.add_subplot(2, max(len(independent_variables), len(dependent_variables)), x)
        if variable in environment.goals['Variable'].to_numpy():
            data_df.plot(x='timestamp', y=[variable, (variable + ' target')], xlabel='timestamp', ylabel=uoms[i],
                kind='line', title=variable, ax=ax)
        else:
            data_df.plot(x='timestamp', y=[variable], xlabel='timestamp', ylabel=uoms[i], kind='line', title=variable, ax=ax)
        i += 1
    pad = 10
    for axis, row in zip([fig.axes[0], fig.axes[len(independent_variables)]], ['MVs / DVs', 'CVs']):
        axis.annotate(row, xy=(0, 0.5), xytext=(-axis.yaxis.labelpad - pad, 0),
                    xycoords=axis.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    fig.subplots_adjust(left=0.15)
    plt.savefig(visuals_file + 'controller_prediction.png')
    plt.show()

def step_by_step_control(environment, move_intervals, independent_variables, raw_state, prediction_models, training_avgs, training_stds):
    """
        --------------------------------------------------------------------------------------------------------
        Title:    Step-by-Step Control
        Author:   Russell Bell
        Date:     August 10, 2021
        --------------------------------------------------------------------------------------------------------
        This program performs calculations for every possible controller action and then returns the action(s)
        that result in optimal controller performance. This controller only accounts for moves that are one step
        out.
        --------------------------------------------------------------------------------------------------------
    """

    all_moves = list(product(range(0, move_intervals), repeat = len(independent_variables[(independent_variables['Max Move'] > 0)])))
    init_state = environment.state
    init_velocity = environment.response_speed
    action_reward = []
    for i in range(move_intervals ** len(independent_variables[(independent_variables['Max Move'] > 0)])):
        environment.state = init_state
        environment.response_speed = init_velocity
        _, reward = environment.step(all_moves[i], move_intervals, raw_state, prediction_models, training_avgs, training_stds)
        action_reward.append(reward)
    environment.state = init_state
    environment.response_speed = init_velocity
    action = action_reward.index(max(action_reward))

    return all_moves[action]

def logical_overrides(environment):
    """
        --------------------------------------------------------------------------------------------------------
        Title:    Logical Overrides
        Author:   Russell Bell
        Date:     August 10, 2021
        --------------------------------------------------------------------------------------------------------
        This program manually sets the controller action(s) based on the presence of predefined conditions. This
        program needs to return the action(s) as a tuple.
        --------------------------------------------------------------------------------------------------------
    """

    action = None

    return action

def controller(period_size, prediction_periods, move_intervals, prediction_start, environment, model_folder, initial_data_file, pred_data_file, visuals_file, run_visuals):

    independent_variables = environment.independent_variables['Variable'].to_numpy()
    independent_variables_uom = environment.independent_variables['UOM'].to_numpy()
    dependent_variables = environment.dependent_variables['Variable'].to_numpy()
    dependent_variables_uom = environment.dependent_variables['UOM'].to_numpy()

    class Progress_screen(threading.Thread):

        def __init__(self, period):
            threading.Thread.__init__(self)
            self.period = period
        def callback(self):
            self.root.quit()
        def run(self):
            self.root = Tk()
            self.root.protocol("WM_DELETE_WINDOW", self.callback)
            self.root.title('Controller')
            label = Label(self.root, text=('Progress: ' + str(self.period) + '/' + str(prediction_periods) + ' periods predicted'))
            label.pack(pady=10)
            self.root.mainloop()

    progress_screen = Progress_screen(period=0)
    progress_screen.start()

    window_size = 1 if environment.independent_variables['Settling Time (periods)'].max() == 0 else environment.independent_variables['Settling Time (periods)'].max()

    # Loading in existing models to be used for prediction and control
    prediction_models = []
    training_avgs = []
    training_stds = []
    for model_num in range(environment.dependent_variables['Model Group'].max()):
        prediction_model = torch.load(model_folder + 'prediction_model_' + str(model_num + 1) + '.pt')
        training_avgs_df = pd.read_csv(model_folder + 'Training Averages - Model_' + str(model_num + 1) + '.csv')
        training_stds_df = pd.read_csv(model_folder + 'Training Standard Deviations - Model_' + str(model_num + 1) + '.csv')
        training_avgs.append(training_avgs_df)
        training_stds.append(training_stds_df)
        prediction_models.append(prediction_model)
    
    # The following "for" loop will simulate the controller actions over the specified prediction periods
    for period in tqdm(range(prediction_periods), "Predicting"):
        if period == 0:
            # For the first prediction period, the initial dataset is used
            data_file = initial_data_file
            data_df = pd.read_csv(initial_data_file)
            data_df = data_df[data_df['timestamp'] < prediction_start]
            data_df.to_csv(pred_data_file, index=False, mode='w')
        else:
            # After the first prediction period, predictions will be based on the previous values
            data_file = pred_data_file
            data_df = pd.read_csv(data_file)
            live_data_df = pd.read_csv(data_file)
        
        init_response_speed = np.zeros(len(environment.dependent_variables))
        state_df = data_df.tail(window_size)
        for var in environment.independent_variables[environment.independent_variables['Model Group'] == model_num+1].loc[:,'Variable']:
            environment.independent_variables.set_index('Variable', inplace=True)
            state_df = state_df.copy()
            window_size = 1 if environment.independent_variables['Settling Time (periods)'][var] == 0 else environment.independent_variables['Settling Time (periods)'][var]
            state_df[var] = state_df[var].shift(environment.independent_variables['Dead Time (periods)'][var])
            state_df[var] = state_df[var].rolling(window_size, min_periods=1, win_type='exponential').mean(tau=-(window_size-1) / np.log(0.01), center=window_size, sym=False)
            environment.independent_variables.reset_index(inplace=True)
        raw_state = data_df.tail(window_size).loc[:, independent_variables].to_numpy()[-1]
        init_state = environment.state = state_df.loc[:, independent_variables].to_numpy()[-1]
        init_response_speed[0] = environment.response_speed[0] = state_df.loc[:,dependent_variables[0]].to_numpy()[-1]

        action = logical_overrides(environment) # Manually specify controller moves based on predefined conditions
        if action == None:
            action = step_by_step_control(environment, move_intervals, environment.independent_variables, raw_state, prediction_models, training_avgs, training_stds)

        environment.state = init_state
        environment.response_speed = init_response_speed
        environment.state, _ = environment.step(action, move_intervals, raw_state, prediction_models, training_avgs, training_stds)

        # state_df['timestamp'] = pd.to_datetime(state_df['timestamp']) # Use if timestamp column is formatted in datetime
        timestamp = round(state_df.iloc[-1, 0] + period_size, 0)

        data = np.concatenate((np.array([timestamp]), environment.state), axis=None)
        data = np.concatenate((data, environment.response_speed), axis=None)

        next_step_df = pd.DataFrame([data], columns=state_df.columns)
        
        next_step_df.to_csv(pred_data_file, index=False, header=None, mode='a')

        progress_screen.period = period

    data_df = pd.read_csv(data_file)
    data_df = data_df.tail(int(prediction_periods * 1.25))
    for i in range(len(environment.goals)):
        target = [environment.goals['Target Value'][i]] * int(prediction_periods * 1.25)
        environment.goals.reset_index(inplace=True)
        data_df[(environment.goals['Variable'][i] + ' target')] = target
        environment.goals.set_index('Variable')
    
    if run_visuals == True:
        control_visualizer(data_df, independent_variables, independent_variables_uom, dependent_variables, dependent_variables_uom, visuals_file, environment)

    return action, data_df

"""
REFERENCES

    Title:          Deep Reinforcement Learning for Process Control: A Primer for Beginners
    Authors:        Steven Spielberga, Aditya Tulsyana, Nathan P. Lawrenceb, Philip D Loewenb, R. Bhushan Gopalunia
    Date:           April 14, 2020
    Availability:   https://arxiv.org/pdf/2004.05490.pdf

    Title:          Non-Linear Tank Level Control for Industrial Applications
    Authors:        Michael Short, A. Arockia Selvakumar
    Date:           September 2020
    Availability:   https://www.scirp.org/journal/paperinformation.aspx?paperid=102677

    Title:          Model-Based Control using Neural Network: A Case Study
    Authors:        Stephen Adhisaputra
    Date:           December 12, 2020
    Availability:   https://medium.com/swlh/model-based-rl-for-nonlinear-dynamics-control-a-case-study-70c31810f255

    Title:          TradingBot
    Authors:        Shivam Akhauri
    Date:           March 3, 2019
    Availability:   https://github.com/shivamakhauri04/TradingBot/blob/master/1_dqn.ipynb

"""