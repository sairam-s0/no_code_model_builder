# main.py

# This is the main entry point for the Universal ML Model Builder application.
# Its primary role is to create the main window and start the GUI application.

import tkinter as tk
# We no longer need to modify the system path here.
# The GUI module will now handle its own dependencies.

# We can now directly import the application class.
from frontend.gui import ModelBuilderApp

def main():
    """
    Initializes the Tkinter root window and runs the main application loop.
    """
    try:
        root = tk.Tk()
        app = ModelBuilderApp(root)
        root.mainloop()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        try:
            from tkinter import messagebox
            messagebox.showerror("Fatal Error", f"A fatal error occurred and the application must close:\n\n{e}")
        except:
            pass

if __name__ == "__main__":
    main()
