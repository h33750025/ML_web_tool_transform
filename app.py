import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.patches as mpatches

# --- Page Setup ---
st.set_page_config(page_title="3D Surface Plotter & ML", layout="wide")

# --- Session State Initialization ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'selected_points' not in st.session_state:
    st.session_state.selected_points = pd.DataFrame()
if 'temp_selected_cache' not in st.session_state:
    st.session_state.temp_selected_cache = {}
if 'model' not in st.session_state:
    st.session_state.model = None
if 'training_meta' not in st.session_state:
    st.session_state.training_meta = {}

# -- Persistence for Page 2 (Training) --
if 'train_loss' not in st.session_state:
    st.session_state.train_loss = None
if 'train_y_true' not in st.session_state:
    st.session_state.train_y_true = None
if 'train_y_pred' not in st.session_state:
    st.session_state.train_y_pred = None

# -- Persistence for Page 3 & 4 (Prediction) --
if 'last_pred_results' not in st.session_state:
    st.session_state.last_pred_results = None

# --- Helper Functions ---

def pick_points_for_temp(temp_val, n_points, df):
    """
    Intelligent sampling logic: 
    Forces inclusion of min/max Frequency points for the given temperature,
    then samples the remaining requested points proportionally.
    """
    sub = df[df["Temperature"] == temp_val]
    if sub.empty:
        return pd.DataFrame()
    
    sub_sorted = sub.sort_values(by="Frequency")
    total_rows = len(sub_sorted)
    
    if total_rows <= n_points:
        return sub_sorted
    
    # Always pick lowest and highest Frequency
    lowest = sub_sorted.iloc[[0]]
    highest = sub_sorted.iloc[[-1]]
    
    if n_points <= 2:
        return pd.concat([lowest, highest]).head(n_points)
    
    n_to_select = n_points - 2
    remaining = sub_sorted.iloc[1:-1]
    
    if remaining.empty:
        return pd.concat([lowest, highest]).drop_duplicates()
        
    # Proportional sampling using indices
    indices_float = np.linspace(0, len(remaining) - 1, n_to_select)
    indices_int = np.unique(indices_float.round().astype(int))
    proportional_part = remaining.iloc[indices_int]
    
    return pd.concat([lowest, highest, proportional_part], ignore_index=True)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to step:", ["1. Load & Sample", "2. Train Model", "3. Prediction", "4. Surface View"])

# ================= PAGE 1: Load & Sample =================
if page == "1. Load & Sample":
    st.title("Step 1: Load Data & Select Points")
    
    # 1. File Uploader
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            # Only read if we haven't already loaded this exact file to avoid reload loops
            # Simple check: if data is None, read it.
            if st.session_state.data is None:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.success(f"Loaded data: {len(st.session_state.data)} rows")
        except Exception as e:
            st.error(f"Error loading file: {e}")

    # 2. Temperature Selection
    if st.session_state.data is not None:
        df = st.session_state.data
        if "Temperature" not in df.columns:
            st.error("CSV must contain a 'Temperature' column.")
        else:
            temps = sorted(df["Temperature"].dropna().unique())
            
            st.write("### Select Temperatures")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                selected_temps = st.multiselect("Choose Temperatures to include:", temps, default=temps)
            
            with col2:
                points_per_temp = st.number_input("Points per Temperature", min_value=2, value=10)
            
            # 3. Action Buttons
            if st.button("Pick Points & Memorize"):
                cache = {}
                for t in selected_temps:
                    sampled = pick_points_for_temp(t, points_per_temp, df)
                    cache[t] = sampled
                
                # Save to session state
                st.session_state.temp_selected_cache = cache
                
                # Flatten into one dataframe
                all_frames = list(cache.values())
                if all_frames:
                    st.session_state.selected_points = pd.concat(all_frames, ignore_index=True)
                    st.success(f"Selected {len(st.session_state.selected_points)} points across {len(cache)} temperatures.")
                else:
                    st.warning("No points selected.")

    # 4. Plotting (Persists because it checks st.session_state.selected_points)
    if not st.session_state.selected_points.empty:
        st.write("### Preview Selection")
        
        # 3D Plot using Matplotlib
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Selected points (Viridis)
        sel = st.session_state.selected_points
        sc = ax.plot_trisurf(sel["Frequency"], sel["Temperature"], sel["Storage Modulus"], 
                             cmap="viridis", edgecolor="none", alpha=1.0)
        
        # Original Data (Ghost/Pink)
        full = st.session_state.data
        if full is not None:
            ax.plot_trisurf(full["Frequency"], full["Temperature"], full["Storage Modulus"], 
                            color='pink', edgecolor='none', alpha=0.1)
        
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_zlabel("Storage Modulus (MPa)")
        
        # --- ROTATION 1 ---
        ax.view_init(elev=30, azim=120)
        
        fig.colorbar(sc, shrink=0.5, aspect=10)
        
        st.pyplot(fig)
        
        with st.expander("View Raw Data"):
            st.dataframe(st.session_state.selected_points)

# ================= PAGE 2: Training =================
elif page == "2. Train Model":
    st.title("Step 2: Train Neural Network")
    
    if st.session_state.selected_points.empty:
        st.warning("Please select points on Page 1 first.")
    else:
        df = st.session_state.selected_points
        cols = df.columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            X_cols = st.multiselect("Input (X) Columns", cols, default=["Frequency", "Temperature"])
        with col2:
            y_cols = st.multiselect("Output (Y) Columns", cols, default=["Storage Modulus"])
            
        epochs = st.number_input("Epochs", min_value=1, value=50)
        
        # --- Training Action ---
        if st.button("Start Training"):
            if not X_cols or not y_cols:
                st.error("Please select both Input and Output columns.")
            else:
                # Prepare data
                train_df = df.sort_values(by=X_cols)
                X = train_df[X_cols].values
                y = train_df[y_cols].values
                
                # Build Model
                model = Sequential([
                    Dense(256, activation='relu', input_shape=(X.shape[1],)),
                    Dense(256, activation='relu'),
                    Dense(len(y_cols), activation='linear')
                ])
                model.compile(optimizer=Adam(0.001), loss='mse', metrics=['root_mean_squared_error'])
                
                # Training UI placeholders
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                class StreamlitCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        p = (epoch + 1) / epochs
                        progress_bar.progress(p)
                        rmse = logs.get("root_mean_squared_error")
                        status_text.text(f"Epoch {epoch+1}/{epochs} - RMSE: {rmse:.4f}")
                
                with st.spinner("Training model..."):
                    history = model.fit(X, y, epochs=epochs, verbose=0, callbacks=[StreamlitCallback()])
                
                # Save Model and Metadata
                st.session_state.model = model
                st.session_state.training_meta = {"X_cols": X_cols, "y_cols": y_cols}
                
                # Save Plotting Data for Persistence
                st.session_state.train_loss = history.history['root_mean_squared_error']
                st.session_state.train_y_true = y
                st.session_state.train_y_pred = model.predict(X)
                
                st.success("Training Complete!")

        # --- Plotting (Persistent) ---
        if st.session_state.train_loss is not None:
            # Plot Training History
            st.subheader("Training Loss (RMSE)")
            st.line_chart(st.session_state.train_loss)
            
            # Plot Pred vs True
            st.subheader("Prediction vs Ground Truth (Training Set)")
            
            y_true = st.session_state.train_y_true
            y_pred = st.session_state.train_y_pred
            
            fig_res, ax_res = plt.subplots()
            ax_res.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.6)
            
            # Diagonal line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax_res.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            ax_res.set_xlabel("Ground Truth")
            ax_res.set_ylabel("Prediction")
            
            r2 = r2_score(y_true.flatten(), y_pred.flatten())
            ax_res.set_title(f"R² Score: {r2:.4f}")
            st.pyplot(fig_res)

# ================= PAGE 3: Prediction =================
elif page == "3. Prediction":
    st.title("Step 3: Prediction on Full Data")
    
    if st.session_state.model is None:
        st.warning("Please train a model on Step 2 first.")
    elif st.session_state.data is None:
        st.warning("No data loaded.")
    else:
        model = st.session_state.model
        meta = st.session_state.training_meta
        
        st.write(f"**Model Inputs:** {meta.get('X_cols')}")
        st.write(f"**Model Outputs:** {meta.get('y_cols')}")
        
        # --- Prediction Action ---
        if st.button("Run Prediction on Full Dataset"):
            full_data = st.session_state.data
            
            try:
                X_full = full_data[meta['X_cols']].values
                y_true = full_data[meta['y_cols']].values.flatten()
                
                y_pred = model.predict(X_full).flatten()
                
                # Use first 2 inputs for axes
                x_col_name = meta['X_cols'][0]
                y_col_name = meta['X_cols'][1] if len(meta['X_cols']) > 1 else "Index"
                
                x_axis = full_data[x_col_name]
                y_axis = full_data[y_col_name] if len(meta['X_cols']) > 1 else np.arange(len(full_data))
                
                # Save results into session state for persistence
                st.session_state.last_pred_results = {
                    "X": x_axis, 
                    "Y": y_axis, 
                    "Z_pred": y_pred, 
                    "Z_true": y_true,
                    "labels": meta,
                    "metrics": {
                        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
                        "r2": r2_score(y_true, y_pred)
                    }
                }
                
            except KeyError as e:
                st.error(f"Missing columns in full dataset: {e}")

        # --- Plotting (Persistent) ---
        if st.session_state.last_pred_results is not None:
            res = st.session_state.last_pred_results
            
            # Show Metrics
            col1, col2 = st.columns(2)
            col1.metric("RMSE", f"{res['metrics']['rmse']:.4f}")
            col2.metric("R² Score", f"{res['metrics']['r2']:.4f}")
            
            # 3D Comparison Plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot Ground Truth
            ax.plot_trisurf(res["X"], res["Y"], res["Z_true"], color='green', alpha=0.3, label='Ground Truth')
            # Plot Prediction
            ax.plot_trisurf(res["X"], res["Y"], res["Z_pred"], color='blue', alpha=0.6, label='Prediction')
            
            ax.set_xlabel(res["labels"]["X_cols"][0])
            ax.set_ylabel(res["labels"]["X_cols"][1] if len(res["labels"]["X_cols"]) > 1 else "Index")
            ax.set_title("Ground Truth (Green) vs Prediction (Blue)")
            
            # Fake legend
            green_patch = mpatches.Patch(color='green', alpha=0.3, label='Ground Truth')
            blue_patch = mpatches.Patch(color='blue', alpha=0.6, label='Prediction')
            ax.legend(handles=[green_patch, blue_patch])
            
            # --- ROTATION 2 ---
            ax.view_init(elev=30, azim=120)
            
            st.pyplot(fig)

# ================= PAGE 4: Surface View =================
elif page == "4. Surface View":
    st.title("Step 4: Predicted Surface Only")
    
    if st.session_state.last_pred_results is None:
        st.warning("Please run a prediction on Step 3 first.")
    else:
        res = st.session_state.last_pred_results
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_trisurf(res["X"], res["Y"], res["Z_pred"], cmap='viridis', edgecolor='none')
        
        x_label = res["labels"]["X_cols"][0]
        y_label = res["labels"]["X_cols"][1] if len(res["labels"]["X_cols"]) > 1 else "Index"
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("ML Predicted Surface")
        
        # --- ROTATION 3 ---
        ax.view_init(elev=30, azim=120)
        
        fig.colorbar(surf, shrink=0.5, aspect=10)
        
        st.pyplot(fig)
        
        # Download button
        csv_df = pd.DataFrame({
            x_label: res["X"], 
            y_label: res["Y"], 
            "Predicted_Z": res["Z_pred"]
        })
        csv = csv_df.to_csv(index=False).encode('utf-8')
        
        st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")