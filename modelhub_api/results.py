from delphi.database import *
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import Tkinter as tk
import ttk
import numpy as np
import pandas as pd

LARGE_FONT= ("Verdana", 12)



class DelphiPerfViewer(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "Delphi Result Viewer")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        self.frames[StartPage] = StartPage(container, self)

        self.frames[StartPage].grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.human_perf_label = ttk.Label(self, text='Show Best Human Performance:')
        self.human_perf_label.grid(row=0,column=0,sticky='E')

        self.human_perf_value = tk.IntVar()
        self.human_perf_value.set(0)
        self.human_perf_button = ttk.Checkbutton(self,variable=self.human_perf_value, command=self.checkbox_change)
        self.human_perf_button.grid(row=0,column=1)

        self.dataset_label = ttk.Label(self, text='Select Dataset:')
        self.dataset_label.grid(row=0, column=2, sticky='E')

        self.dataset_selector = ttk.Combobox(self, values=self.get_dataset_name_list(), width=30, state='readonly')
        self.dataset_selector.bind("<<ComboboxSelected>>", self.new_dataset_selection)
        self.dataset_selector.grid(row=0, column=3)

        self.f = Figure(figsize=(8, 4.5), dpi=100)
        self.a = self.f.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.f, self)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=4)

        # toolbar = NavigationToolbar2TkAgg(canvas, self)
        # toolbar.update()
        # canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def new_dataset_selection(self, event):
        self.plot()

    def checkbox_change(self):
        self.plot()

    def plot(self):
        self.a.clear()

        selection = self.dataset_selector.get()
        dataset_name = selection.replace('(', ',').replace(')', ',').split(',')[0].strip()
        dataset_id = selection.replace('(', ',').replace(')', ',').split(',')[1]

        learners = GetLearners(datarun_id=dataset_id)

        values = []

        for learner in learners:
            if learner.is_error == 0:
                values.append(learner.cv)

        self.a.plot(values, label='Delphi Perf')

        if self.human_perf_value.get() == 1:
            val = self.get_best_human_perf(dataset_name)
            line_values = val * np.ones(len(values))
            self.a.plot(line_values, label='Best Human Perf')

        self.a.set_title('{} performance over time'.format(self.dataset_selector.get()))
        self.a.set_xlabel('Classifier (Sorted by Start Time)')
        self.a.set_ylabel('Accuracy')
        self.a.legend(loc='best')

        self.canvas.draw()

    def get_dataset_name_list(self):
        dataruns = GetAllDataruns()

        list = []

        for run in dataruns:
            list.append('{} ({})'.format(run.name, run.id))

        return list

    def get_best_human_perf(self, dataset_name):
        openml_filename_to_id_file = '/Users/tss/Dropbox/MITvisit/OpenMLIDandFilenames.csv'
        human_perf_file = '/Users/tss/Dropbox/MITvisit/FilesFromWeilian/openml_best_runs.csv'

        f2imap = pd.read_csv(openml_filename_to_id_file)
        entry = f2imap.loc[f2imap['filename'] == ''.join((dataset_name,'.csv'))]

        openml_id = int(entry['ID'].values[0])

        info = pd.read_csv(human_perf_file)
        entry = info.loc[info['Dataset ID'] == openml_id]

        return float(entry['Accuracy'].values[0])


app = DelphiPerfViewer()
app.mainloop()