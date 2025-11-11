import os
from os import environ, environb
from tkinter import *
from tkinter import ttk
from MPC import model_training_script, controller
from importlib.machinery import SourceFileLoader

project_names = []
projects = []
for root, dirs, files in os.walk(os.getcwd()):
    for name in files:
        if name.endswith('Environment.py'):
            project = root.replace((os.getcwd() + '/Projects/'), '')
            project = project.replace(('/Model Information'), '')
            project_names.append(project)
            module = SourceFileLoader("Environment", (root + '/Environment.py')).load_module()
            projects.append(module.environment)

def modeller_gui(project_name, environment):

    root_folder = os.getcwd() + '/Projects/' + project_name
    model_folder = root_folder + '/Models/'

    modelling_visuals_folder = root_folder + '/Plots/Modelling/'
    historical_data_file = root_folder + '/Data/hist_training.csv'

    environment = environment()

    fields = ('Epochs',)
    def makeform(root, fields):
        entries = {}
        for field in fields:
            row = Frame(root)
            lab = Label(row, width=22, text=field+": ", anchor='w')
            ent = Entry(row)
            ent.insert(0,"0")
            row.pack(side = TOP, fill = X, padx = 5 , pady = 5)
            lab.pack(side = LEFT)
            ent.pack(side = RIGHT, expand = YES, fill = X)
            entries[field] = ent
        return entries
    
    root = Tk()
    root.title('Modeller')
    ents = makeform(root, fields)
    root.bind('<Return>', (lambda event, e = ents: e))
    b1 = Button(root, text = 'Begin Training',
        command=(lambda e = ents: model_training_script(historical_data_file, model_folder, modelling_visuals_folder, 
            environment.independent_variables, environment.dependent_variables, int(ents['Epochs'].get()), show_visuals=False)))
    b1.pack(side = LEFT, padx = 5, pady = 5)
    b3 = Button(root, text = 'Quit', command = root.quit)
    b3.pack(side = LEFT, padx = 5, pady = 5)
    root.mainloop()

def controller_gui(project_name, environment):
    
    root_folder = os.getcwd() + '/Projects/' + project_name
    model_folder = root_folder + '/Models/'
    initial_data_file = root_folder + '/Data/initial_data.csv'
    pred_data_file = root_folder + '/Data/pred_data.csv'
    visuals_file = root_folder + '/Plots/Control/'

    environment = environment()

    fields = ('Period Size', 'Prediction Periods', 'Moves Considered per Step', 'Starting Timestamp')
    def makeform(root, fields):
        entries = {}
        for field in fields:
            row = Frame(root)
            lab = Label(row, width=22, text=field+": ", anchor='w')
            ent = Entry(row)
            ent.insert(0,"0")
            row.pack(side = TOP, fill = X, padx = 5 , pady = 5)
            lab.pack(side = LEFT)
            ent.pack(side = RIGHT, expand = YES, fill = X)
            entries[field] = ent
        return entries
    
    root = Tk()
    root.title('Controller')
    ents = makeform(root, fields)
    root.bind('<Return>', (lambda event, e = ents: e))
    b1 = Button(root, text = 'Run Scenario',
        command = (lambda e = ents: controller(float(ents['Period Size'].get()), int(ents['Prediction Periods'].get()), 
            int(ents['Moves Considered per Step'].get()), float(ents['Starting Timestamp'].get()), environment, 
                model_folder, initial_data_file, pred_data_file, visuals_file, run_visuals=True)))
    b1.pack(side = LEFT, padx = 5, pady = 5)
    b3 = Button(root, text = 'Quit', command = root.quit)
    b3.pack(side = LEFT, padx = 5, pady = 5)
    root.mainloop()

def run():
    if v.get() == "1":
        modeller_gui(dropdown.get(), projects[project_names.index(dropdown.get())])
    else:
        controller_gui(dropdown.get(), projects[project_names.index(dropdown.get())])

root = Tk()
root.title('Model Predictive Controller')
ttk.Label(root, text = "Select the Project:").pack()
selected_project = StringVar()
dropdown = ttk.Combobox(root, width=23, textvariable=selected_project)
dropdown['values'] = tuple(project_names)
dropdown.pack(pady=5)

ttk.Label(root, text = "Select Program:").pack()
v = StringVar(root, "1")
values = {"Model Trainer" : "1",
        "Controller" : "2"}
for (text, value) in values.items():
    Radiobutton(root, text = text, variable = v,
        value = value).pack(side = TOP, ipady = 2.5)

submit_button = Button(root, text = 'Submit', command=run).pack(side = LEFT, padx = 5, pady = 5)
quit_button = Button(root, text = 'Quit', command = root.quit).pack(side = LEFT, padx = 5, pady = 5)

root.mainloop()