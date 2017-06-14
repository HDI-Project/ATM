from delphi.database import *
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import Tkinter as tk
import ttk
import re

LARGE_FONT= ("Verdana", 12)

def get_dataset_name_list():
    dataruns = GetAllDataruns()

    list = []

    for run in dataruns:
        list.append('{} ({})'.format(run.name,run.id))

    return list

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
        tk.Frame.__init__(self,parent)

        self.dataset_selector = ttk.Combobox(self, values=get_dataset_name_list(), width=50, state='readonly')
        self.dataset_selector.bind("<<ComboboxSelected>>", self.new_selection)
        self.dataset_selector.pack()


        self.f = Figure(figsize=(10, 5), dpi=100)
        self.a = self.f.add_subplot(111)


        self.canvas = FigureCanvasTkAgg(self.f, self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # toolbar = NavigationToolbar2TkAgg(canvas, self)
        # toolbar.update()
        # canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def new_selection(self, event):
        self.a.clear()

        selection = self.dataset_selector.get()
        dataset_id = selection.replace('(', ',').replace(')', ',').split(',')[1]

        learners = GetLearners(datarun_id=dataset_id)

        values = []

        for learner in learners:
            if learner.is_error == 0:
                values.append(learner.cv)

        self.a.plot(values)
        self.a.set_title('{} performance over time'.format(self.dataset_selector.get()))
        self.a.set_xlabel('Classifier (Sorted by Start Time)')
        self.a.set_ylabel('Accuracy')

        self.canvas.draw()



app = DelphiPerfViewer()
app.mainloop()