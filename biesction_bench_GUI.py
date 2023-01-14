from tkinter import ttk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from GraPart.benchmarks import parallel_benchmark_bisection
from GraPart.stat_generator import generate_graphs_for_Bisection
import time

class MyApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Bench mark for bisection")
        self.input_frame = tk.Frame(self)

        # Create the numeric inputs
        self.input1_label = tk.Label(self.input_frame, text="#Nodes:")
        self.input1 = tk.Entry(self.input_frame)
        self.input1.insert(0, 400)
        self.input2_label = tk.Label(self.input_frame, text="xMax:")
        self.input2 = tk.Entry(self.input_frame)
        self.input2.insert(0, 20)
        self.input3_label = tk.Label(self.input_frame, text="yMax:")
        self.input3 = tk.Entry(self.input_frame)
        self.input3.insert(0, 20)
        self.input4_label = tk.Label(
            self.input_frame, text="Connect threshold:")
        self.input4 = tk.Entry(self.input_frame)
        self.input4.insert(0, 1)
        self.input5_label = tk.Label(self.input_frame, text="#Max Clusters:")
        self.input5 = tk.Entry(self.input_frame)
        self.input5.insert(0, 30)
        self.input7_label = tk.Label(self.input_frame, text="#Max firewalls:")
        self.input7 = tk.Entry(self.input_frame)
        self.input7.insert(0, 50)
        self.input6_label = tk.Label(self.input_frame, text="#Max iteration:")
        self.input6 = tk.Entry(self.input_frame)
        self.input6.insert(0, 1000)
        self.input10_label = tk.Label(self.input_frame, text="Margin (for Oneway):")
        self.input10 = tk.Entry(self.input_frame)
        self.input10.insert(0, 0.1)
        self.input9_label = tk.Label(self.input_frame, text="Save directory:")
        self.input9 = tk.Entry(self.input_frame)
        self.input9.insert(0, f'date_{time.time()}')
        
      # Create a variable to hold the toggle state
        self.input8_label = tk.Label(
            self.input_frame, text="Firewall variation:")
        self.toggle_var = tk.StringVar()
        self.toggle_var.set("self")  # set default value

        # Create the toggle button
        self.toggle_button1 = tk.Radiobutton(
            self.input_frame, text="self", variable=self.toggle_var, value="self")
        self.toggle_button2 = tk.Radiobutton(
            self.input_frame, text="other", variable=self.toggle_var, value="other")

        # Place the input frame in the window
        self.input_frame.pack(side="top")

        # Create the button
        self.button = tk.Button(
            self.input_frame, text="Run algorithm", command=self.run_algorithm)

        # Place the widgets in the frame
        self.input1_label.pack(side="top")
        self.input1.pack(side="top")
        self.input2_label.pack(side="top")
        self.input2.pack(side="top")
        self.input3_label.pack(side="top")
        self.input3.pack(side="top")
        self.input4_label.pack(side="top")
        self.input4.pack(side="top")
        self.input5_label.pack(side="top")
        self.input5.pack(side="top")
        self.input6_label.pack(side="top")
        self.input6.pack(side="top")
        self.input7_label.pack(side="top")
        self.input7.pack(side="top")
        self.input10_label.pack(side="top")
        self.input10.pack(side="top") 
        self.input8_label.pack(side="top")
        self.toggle_button1.pack(side="top")
        self.toggle_button2.pack(side="top")
        self.input9_label.pack(side="top")
        self.input9.pack(side="top")

        self.button.pack(side="top")
        # Create the labels for the input fields

        # Place the labels in the frame, to the top of the corresponding input field

        # Place the input frame in the window
        self.input_frame.pack(side="left")

        # Create a frame to hold the Notebook widget
        self.output_frame = tk.Frame(self)
        self.output_frame.pack(side="top")

    def run_algorithm(self):

        self.notebook = ttk.Notebook(self.output_frame)
        # Get the values of the numeric inputs
        num_nodes = int(self.input1.get())
        xMax = int(self.input2.get())
        yMax = int(self.input3.get())
        connect_threshold = float(self.input4.get())
        max_clusters = int(self.input5.get())
        max_firewalls = int(self.input7.get())
        max_iterations = int(self.input6.get())
        variation = self.toggle_var.get()
        save_dir = self.input9.get()
        margin = float(self.input10.get())

        self.show_output("Running algorithm...", self.output_frame)
        # Run the algorithm using the input values and selected option
        parallel_benchmark_bisection(max_clusters,max_firewalls, num_nodes, connect_threshold, xMax, yMax, variation, max_iter=max_iterations,margin=margin, save_dir=save_dir)
        generate_graphs_for_Bisection(save_dir)
        self.show_output("Algorithm finished", self.output_frame)

    def show_output(self, output, frame):
        # Create a Label widget to display the output value
        label = tk.Label(frame, text=output)
        # Pack the widget to display it in the frame
        label.pack()

        


if __name__ == "__main__":
    app = MyApp()
    app.mainloop()
