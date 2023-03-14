from tkinter import ttk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from GraPart.showcase import single_network_showcase, bisection_showcase

class MyApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Bisection Showcase")
        # Create a frame to hold the input widgets
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
        self.input4_label = tk.Label(self.input_frame, text="Connect threshold:")
        self.input4 = tk.Entry(self.input_frame)
        self.input4.insert(0, 1)
        self.input5_label = tk.Label(self.input_frame, text="#Max Clusters:")
        self.input5 = tk.Entry(self.input_frame)
        self.input5.insert(0, 30)
        self.input7_label = tk.Label(self.input_frame, text="#Max firewalls:")
        self.input7 = tk.Entry(self.input_frame)
        self.input7.insert(0, 50)
        self.input8_label = tk.Label(self.input_frame, text="#Max iterations:")
        self.input8 = tk.Entry(self.input_frame)
        self.input8.insert(0, 100)
        self.input10_label = tk.Label(self.input_frame, text="Margin (for Oneway):")
        self.input10 = tk.Entry(self.input_frame)
        self.input10.insert(0, 0.1)
        self.input6_label = tk.Label(self.input_frame, text="Firewall variation:")
        # Create the radio buttons
        # Create a variable to hold the toggle state
        self.toggle_var = tk.StringVar()
        self.toggle_var.set("other") # set default value

        # Create the toggle button
        self.toggle_button1 = tk.Radiobutton(self.input_frame, text="self", variable=self.toggle_var, value="self")
        self.toggle_button2 = tk.Radiobutton(self.input_frame, text="other", variable=self.toggle_var, value="other")
      

        # Place the input frame in the window
        self.input_frame.pack(side="top") 


        # Create the button
        self.button = tk.Button(self.input_frame, text="Run algorithm", command=self.run_algorithm)
        self.button2 = tk.Button(self.input_frame, text="Clear", command=self.clear)

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
        self.input7_label.pack(side="top")
        self.input7.pack(side="top")
        self.input8_label.pack(side="top")
        self.input8.pack(side="top")
        self.input10_label.pack(side="top")
        self.input10.pack(side="top")
        self.input6_label.pack(side="top")
        self.toggle_button1.pack(side="top")
        self.toggle_button2.pack(side="top")
        self.button.pack(side="top")
        self.button2.pack(side="top")
       
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
        variation = self.toggle_var.get()
        max_firewalls = int(self.input7.get())
        max_iterations = int(self.input8.get())
        margin = float(self.input10.get())

        # Get the value of the selected radio button
        # selected_option = 1 if self.radio_button1.get() else 2

        # Run the algorithm using the input values and selected option
        results, outputs = bisection_showcase(num_nodes = num_nodes,
                                            max_clusters = max_clusters,
                                            max_firewalls = max_firewalls,
                                            connect_threshold = connect_threshold,
                                            xMax = xMax,
                                            yMax = yMax,
                                            variation = variation,
                                            max_iter = max_iterations,
                                            margin = margin)

        figures, outputs = outputs[0],outputs[1]                                
        matplot_figure1 = figures[0]
        matplot_figure2 = figures[1]
        matplot_figure3 = figures[2]
        output1 = outputs[0]
        output2 = outputs[1]
        output3 = outputs[2]

        # Create the Notebook widget
        self.notebook = ttk.Notebook(self.output_frame)

        # Create three frames to hold the figures and output values
        self.frame1 = tk.Frame(self.notebook)
        self.frame2 = tk.Frame(self.notebook)
        self.frame3 = tk.Frame(self.notebook)
        self.frame4= tk.Frame(self.input_frame)

        # Add the frames to the Notebook widget
        self.notebook.add(self.frame1, text="Figure 1")
        self.notebook.add(self.frame2, text="Figure 2")
        self.notebook.add(self.frame3, text="Figure 3")

        # Show the matplotlib figures in the frames
        self.show_figure(matplot_figure1, self.frame1)
        self.show_figure(matplot_figure2, self.frame2)
        self.show_figure(matplot_figure3, self.frame3)

        # Show the output values in the frames
        self.show_output(output1, self.frame1)
        self.show_output(output2, self.frame2)
        self.show_output(output3, self.frame3)

        results_str = ""
        for i in results:
            results_str += "Clusters: "+ str(i['clusters']) + "Firewalls: "+ str(i['firewalls']) + '\n'
        
        self.show_output(results_str, self.frame4)
        # Pack the Notebook widget
        self.frame4.pack()
        self.notebook.pack()

    def clear(self):
        try:
            # Remove the previous Notebook and frames
            self.notebook.destroy()
            self.frame1.destroy()
            self.frame2.destroy()
            self.frame3.destroy()
            self.frame4.destroy()
        except:
            pass

    def show_figure(self, figure, frame):
        # Create a FigureCanvasTkAgg widget to display the figure
        canvas = FigureCanvasTkAgg(figure, frame)
        canvas.draw()

        # Pack the widget to display it in the frame
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def show_output(self, output, frame):
        # Create a Label widget to display the output value
        label = tk.Label(frame, text=output)

        # Pack the widget to display it in the frame
        label.pack()

if __name__ == "__main__":
    app = MyApp()
    app.mainloop()