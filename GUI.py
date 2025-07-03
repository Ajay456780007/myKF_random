import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import joblib
from tkinter import ttk

# Placeholder feature lists for datasets
CICIDS2015_FEATURES = []
NBAIOT_FEATURES = []
UNSWNB15_FEATURES = []

MODEL_PATHS = {
    'CICIDS2015': 'Saved model/CICIDS2015_model.pkl',
    'N-BaIoT': 'Saved model/N-BaIoT_model.pkl',
    'UNSW-NB15': 'Saved model/UNSW-NB15_model.pkl'
}

class AttackPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚨 Attack Detection System")
        self.root.geometry("800x700")
        self.root.configure(bg="#f0f4f8")

        self.current_features = []
        self.model = None
        self.selected_features = [tk.StringVar() for _ in range(10)]
        self.data_row = None

        # Styling
        title_font = ("Helvetica", 20, "bold")
        label_font = ("Arial", 12)
        button_font = ("Arial", 10, "bold")
        self.bg_color = "#e8f0fe"
        self.highlight_color = "#1e88e5"

        # Title
        tk.Label(root, text="Intrusion Detection System", font=title_font, bg="#f0f4f8", fg="#0d47a1").pack(pady=10)

        # Frame for dataset buttons
        dataset_frame = tk.Frame(root, bg=self.bg_color, bd=2, relief="ridge")
        dataset_frame.pack(pady=10, padx=20, fill="x")

        tk.Label(dataset_frame, text="Select Dataset:", font=label_font, bg=self.bg_color).pack(pady=5)

        btn_frame = tk.Frame(dataset_frame, bg=self.bg_color)
        btn_frame.pack()

        for dataset in MODEL_PATHS.keys():
            ttk.Button(btn_frame, text=dataset, command=lambda d=dataset: self.load_dataset(d)).pack(side="left", padx=10, pady=5)

        # Frame for feature selectors
        feature_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove")
        feature_frame.pack(pady=10, padx=20)

        tk.Label(feature_frame, text="Select 10 Features:", font=label_font, bg="#ffffff").grid(row=0, column=0, columnspan=3, pady=5)

        self.dropdown_menus = []
        for i in range(10):
            tk.Label(feature_frame, text=f"Feature {i+1}:", bg="#ffffff", font=label_font).grid(row=i+1, column=0, sticky="e", padx=5, pady=3)
            dropdown = tk.OptionMenu(feature_frame, self.selected_features[i], '')
            dropdown.config(width=30)
            dropdown.grid(row=i+1, column=1, columnspan=2, pady=3)
            self.dropdown_menus.append(dropdown)

        # Action buttons
        button_frame = tk.Frame(root, bg="#f0f4f8")
        button_frame.pack(pady=20)

        self.select_data_button = tk.Button(button_frame, text="📂 Select Data", font=button_font,
                                            command=self.select_data, state='disabled', bg="#64b5f6", fg="white", width=15)
        self.select_data_button.pack(side="left", padx=20)

        self.predict_button = tk.Button(button_frame, text="🔍 Predict", font=button_font,
                                        command=self.predict, state='disabled', bg="#43a047", fg="white", width=15)
        self.predict_button.pack(side="left", padx=20)

        # Prediction result label
        self.result_label = tk.Label(root, text="", font=("Arial", 16, "bold"), bg="#f0f4f8")
        self.result_label.pack(pady=30)

    def load_dataset(self, dataset_name):
        try:
            self.model = joblib.load(MODEL_PATHS[dataset_name])
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Could not load model for {dataset_name}\n{e}")
            return

        # Load features
        if dataset_name == 'CICIDS2015':
            self.current_features = CICIDS2015_FEATURES
        elif dataset_name == 'N-BaIoT':
            self.current_features = NBAIOT_FEATURES
        elif dataset_name == 'UNSW-NB15':
            self.current_features = UNSWNB15_FEATURES

        # Update dropdowns
        for var, menu in zip(self.selected_features, self.dropdown_menus):
            var.set('')
            menu['menu'].delete(0, 'end')
            for feat in self.current_features:
                menu['menu'].add_command(label=feat, command=tk._setit(var, feat))

        self.select_data_button.config(state='normal')
        self.result_label.config(text="Dataset loaded. Please select features.", fg="#0d47a1")

    def select_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            messagebox.showerror("File Error", f"Failed to load CSV file.\n{e}")
            return

        try:
            selected_columns = [var.get() for var in self.selected_features]
            if '' in selected_columns:
                raise ValueError("All 10 features must be selected.")

            self.data_row = df[selected_columns].iloc[0].values.reshape(1, -1)
            self.predict_button.config(state='normal')
            self.result_label.config(text="✅ Data selected, ready to predict.", fg="#1565c0")
        except Exception as e:
            messagebox.showerror("Data Selection Error", str(e))

    def predict(self):
        if self.model is None or self.data_row is None:
            messagebox.showwarning("Missing Info", "Please load a dataset and select data.")
            return
        try:
            prediction = self.model.predict(self.data_row)[0]
            if prediction in ['attack', 1, 'Attack', 'malicious', 'DoS', 'Botnet']:
                self.result_label.config(text="🚨 Attack Data Detected!", fg="red")
            else:
                self.result_label.config(text="✅ Normal Data", fg="green")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))


# --- Run the App ---
if __name__ == "__main__":
    root = tk.Tk()
    app = AttackPredictorGUI(root)
    root.mainloop()
