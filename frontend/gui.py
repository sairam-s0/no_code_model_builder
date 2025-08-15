import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue

# Add the project root to sys.path dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# It's good practice to handle potential import errors
try:
    from backend import model_loader, dataset_utils, train
except ImportError as e:
    messagebox.showerror("Import Error", f"Could not import backend modules:\n{e}")
    exit()

class ModelBuilderApp:
    """
    The main class for the Tkinter GUI application.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Universal ML Model Builder")
        self.root.geometry("600x750")
        self.root.configure(bg='#f0f0f0')

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', padding=6, relief="flat", background="#0078D7", foreground="white")
        self.style.map('TButton', background=[('active', '#005a9e')])
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TCombobox', font=('Arial', 10))
        self.style.configure('Header.TLabel', font=('Arial', 14, 'bold'))

        # Queue for backend-to-frontend communication
        self.log_queue = queue.Queue()

        self.create_widgets()
        self.process_log_queue()

    def create_widgets(self):
        """Creates and lays out all the GUI widgets."""
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # --- Model Selection ---
        model_frame = ttk.LabelFrame(main_frame, text="1. Model Configuration", padding="10")
        model_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(model_frame, text="Select Model Type:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.model_type = tk.StringVar(value="YOLO")
        ttk.Combobox(model_frame, textvariable=self.model_type, values=["YOLO", "BERT", "Whisper"], width=25).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        # --- Dataset Selection ---
        dataset_frame = ttk.LabelFrame(main_frame, text="2. Dataset Configuration", padding="10")
        dataset_frame.pack(fill=tk.X, pady=10)

        ttk.Label(dataset_frame, text="Select Dataset Source:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.dataset_source = tk.StringVar(value="Built-in")
        source_combo = ttk.Combobox(dataset_frame, textvariable=self.dataset_source, values=["Built-in", "Kaggle", "Custom"], width=25)
        source_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        source_combo.bind("<<ComboboxSelected>>", self.update_dataset_options)

        # Dynamic dataset options frame
        self.dataset_options_frame = ttk.Frame(dataset_frame)
        self.dataset_options_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=tk.W)

        # Built-in dataset dropdown
        self.built_in_dataset = tk.StringVar()
        self.built_in_combo = ttk.Combobox(self.dataset_options_frame, textvariable=self.built_in_dataset, values=["MNIST", "CIFAR-10", "COCO", "IMDB"], width=25)
        
        # Kaggle dataset entry
        self.kaggle_frame = ttk.Frame(self.dataset_options_frame)
        ttk.Label(self.kaggle_frame, text="Kaggle Dataset ID:").pack(side="left", padx=5)
        self.kaggle_dataset_id = tk.StringVar(value="username/dataset-name")
        ttk.Entry(self.kaggle_frame, textvariable=self.kaggle_dataset_id, width=30).pack(side="left")

        # Custom dataset browse
        self.custom_frame = ttk.Frame(self.dataset_options_frame)
        self.dataset_path = tk.StringVar()
        ttk.Entry(self.custom_frame, textvariable=self.dataset_path, width=30, state='readonly').pack(side="left", padx=5)
        self.browse_btn = ttk.Button(self.custom_frame, text="Browse...", command=self.select_custom_dataset)
        self.browse_btn.pack(side="left")

        # --- Training Preferences ---
        training_frame = ttk.LabelFrame(main_frame, text="3. Training Preferences", padding="10")
        training_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(training_frame, text="Epochs:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.epochs = tk.IntVar(value=10)
        ttk.Entry(training_frame, textvariable=self.epochs, width=10).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(training_frame, text="Learning Rate:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.learning_rate = tk.DoubleVar(value=0.001)
        ttk.Entry(training_frame, textvariable=self.learning_rate, width=10).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)

        ttk.Label(training_frame, text="Optimizer:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.optimizer = tk.StringVar(value="Adam")
        ttk.Combobox(training_frame, textvariable=self.optimizer, values=["Adam", "SGD"], width=10).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        self.fine_tune = tk.BooleanVar(value=True)
        ttk.Checkbutton(training_frame, text="Fine-Tune Pretrained Model", variable=self.fine_tune).grid(row=2, column=0, columnspan=2, pady=5, sticky=tk.W)

        # --- Optimization ---
        opt_frame = ttk.LabelFrame(main_frame, text="4. Post-Training Optimization", padding="10")
        opt_frame.pack(fill=tk.X, pady=10)

        self.prune = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text="Apply Pruning (Reduce Size)", variable=self.prune).pack(anchor=tk.W)
        
        self.quantize = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text="Apply Quantization (Increase Speed)", variable=self.quantize).pack(anchor=tk.W)

        # --- Action Button ---
        self.train_button = ttk.Button(main_frame, text="Start Training Pipeline", command=self.start_training_thread)
        self.train_button.pack(pady=20, fill=tk.X, ipady=5)

        # --- Status Log ---
        log_frame = ttk.LabelFrame(main_frame, text="Status Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.log_area = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10, state='disabled', bg='#ffffff')
        self.log_area.pack(fill=tk.BOTH, expand=True)

        # Initial UI state
        self.update_dataset_options()

    def log_message(self, message):
        """Thread-safe way to add a message to the log area."""
        self.log_area.configure(state='normal')
        self.log_area.insert(tk.END, message + '\n')
        self.log_area.configure(state='disabled')
        self.log_area.see(tk.END)

    def process_log_queue(self):
        """Checks the queue for messages from the backend and displays them."""
        try:
            message = self.log_queue.get_nowait()
            self.log_message(message)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_log_queue)

    def select_custom_dataset(self):
        folder = filedialog.askdirectory()
        if folder:
            self.dataset_path.set(folder)

    def update_dataset_options(self, event=None):
        """Shows/hides dataset options based on the selected source."""
        # Hide all widgets first
        self.built_in_combo.pack_forget()
        self.kaggle_frame.pack_forget()
        self.custom_frame.pack_forget()

        source = self.dataset_source.get()
        if source == "Built-in":
            self.built_in_combo.pack(pady=5, anchor=tk.W)
        elif source == "Kaggle":
            self.kaggle_frame.pack(pady=5, anchor=tk.W)
        elif source == "Custom":
            self.custom_frame.pack(pady=5, anchor=tk.W)

    def start_training_thread(self):
        """
        Validates user input and starts the backend pipeline in a separate thread
        to keep the GUI responsive.
        """
        # --- Input Validation ---
        source = self.dataset_source.get()
        dataset_info = ""
        if source == "Built-in":
            if not self.built_in_dataset.get():
                messagebox.showerror("Error", "Please select a built-in dataset!")
                return
            dataset_info = self.built_in_dataset.get()
        elif source == "Kaggle":
            if not self.kaggle_dataset_id.get().strip() or self.kaggle_dataset_id.get() == "username/dataset-name":
                messagebox.showerror("Error", "Please enter a valid Kaggle dataset ID!")
                return
            dataset_info = self.kaggle_dataset_id.get().strip()
        elif source == "Custom":
            if not self.dataset_path.get():
                messagebox.showerror("Error", "Please select a custom dataset folder!")
                return
            dataset_info = self.dataset_path.get()
        
        # --- Collect all parameters ---
        params = {
            "model_choice": self.model_type.get(),
            "dataset_source": source,
            "dataset_info": dataset_info,
            "epochs": self.epochs.get(),
            "lr": self.learning_rate.get(),
            "optimizer_type": self.optimizer.get(),
            "fine_tune": self.fine_tune.get(),
            "prune": self.prune.get(),
            "quantize": self.quantize.get()
        }

        # Disable button to prevent multiple runs
        self.train_button.config(state="disabled", text="Training in Progress...")
        
        # Run the backend pipeline in a daemon thread
        threading.Thread(target=self.run_backend_pipeline, args=(params,), daemon=True).start()

    def run_backend_pipeline(self, params):
        """
        The target function for the training thread. It calls the backend modules.
        """
        try:
            # --- Status Callback Function ---
            # This function will be passed to the backend so it can send updates
            def status_callback(message):
                self.log_queue.put(message)

            status_callback("--- Starting Backend Pipeline ---")
            
            # 1. Load Model
            model = model_loader.load_model(params["model_choice"], status_callback)
            
            # 2. Load Dataset
            dataset_path = dataset_utils.load_dataset(params["dataset_source"], params["dataset_info"], status_callback)
            
            # 3. Train and Optimize
            train.run_training_pipeline(model, dataset_path, params, status_callback)
            
            status_callback("--- Backend Pipeline Finished Successfully! ---")
            self.log_queue.put("SUCCESS")

        except Exception as e:
            self.log_queue.put(f"ERROR: {e}")
        finally:
            # Re-enable the button from the main thread
            self.root.after(0, self.finalize_training)

    def finalize_training(self):
        """
        This method is called on the main thread after training finishes
        to safely update the GUI.
        """
        # Check for success/error messages from the queue
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
                if msg == "SUCCESS":
                     messagebox.showinfo("Success", "Model training pipeline completed successfully!")
                elif msg.startswith("ERROR:"):
                     messagebox.showerror("Pipeline Error", msg)
                else: # regular log message
                    self.log_message(msg)
            except queue.Empty:
                break

        self.train_button.config(state="normal  ", text="Start Training Pipeline")


if __name__ == "__main__":
    root = tk.Tk()
    app = ModelBuilderApp(root)
    root.mainloop()
