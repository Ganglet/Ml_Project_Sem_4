import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import regression models and evaluation metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Set the style for plots
sns.set_style("whitegrid")

def main():
    st.title("IMDb Movie Ratings Dashboard")
    st.write(
        """
        This Streamlit app allows you to explore IMDb movie ratings and run a regression analysis.
        Upload your CSV file to view visualizations, filter movies by rating, and then perform regression.
        In this model, the IMDb rating is treated as a continuous target variable to be predicted.
        
        **Step 1:** Select one or more feature columns (for example, Votes, Year) to predict the IMDb rating.  
        **Step 2:** Choose the regression algorithm:
         - **Linear Regression**
         - **Decision Tree Regressor**
         - **Random Forest Regressor**
         
        The app will then display evaluation metrics including Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.
        """
    )

    # ----------------------------
    # Sidebar: File uploader and options
    # ----------------------------
    st.sidebar.header("Upload & Options")
    uploaded_file = st.sidebar.file_uploader("Upload IMDb Movie Ratings CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error loading the file: {e}")
            return
        
        if df is not None:
            # ----------------------------
            # Data Preview and Basic Information
            # ----------------------------
            st.subheader("Data Preview")
            st.dataframe(df.head())
            st.write("### Basic Data Information")
            st.write(f"**Number of rows:** {df.shape[0]}")
            st.write(f"**Number of columns:** {df.shape[1]}")
            st.write("### Descriptive Statistics")
            st.write(df.describe(include="all"))
            
            # ----------------------------
            # Sidebar Column Selection for Ratings & Additional Feature(s)
            # ----------------------------
            st.sidebar.subheader("Column Selection")
            all_columns = df.columns.tolist()
            rating_col = st.sidebar.selectbox(
                "Select the column for IMDb ratings (target variable)",
                all_columns,
                help="Choose the column in your CSV that represents the IMDb rating."
            )
            vote_col = st.sidebar.selectbox(
                "Select the column for votes (optional)",
                ["(None)"] + all_columns,
                help="Choose the column that represents the number of votes, if available."
            )
            
            # Convert the rating column to numeric and filter out rows that cannot be converted.
            df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')
            num_invalid = df[rating_col].isna().sum()
            if num_invalid > 0:
                st.warning(f"{num_invalid} non-numeric entries in '{rating_col}' were removed.")
                df = df.dropna(subset=[rating_col])
            
            if df[rating_col].empty or df[rating_col].isna().all():
                st.error(f"No valid numeric ratings found in column '{rating_col}'. Please check your data.")
                return
            
            # ----------------------------
            # Data Visualization Section
            # ----------------------------
            st.write("## Distribution of IMDb Ratings")
            fig, ax = plt.subplots()
            sns.histplot(df[rating_col], bins=20, kde=True, ax=ax)
            ax.set_xlabel("IMDb Rating")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            
            # Optional: Scatter Plot for Votes vs Ratings
            if vote_col != "(None)":
                df[vote_col] = pd.to_numeric(df[vote_col], errors='coerce')
                before_drop = df.shape[0]
                df = df.dropna(subset=[vote_col])
                after_drop = df.shape[0]
                if before_drop != after_drop:
                    st.warning("Some rows were dropped due to non-numeric votes.")
                st.write("## Relationship between Votes and Ratings")
                fig2, ax2 = plt.subplots()
                sns.scatterplot(x=df[vote_col], y=df[rating_col], ax=ax2)
                ax2.set_xlabel("Votes")
                ax2.set_ylabel("IMDb Rating")
                st.pyplot(fig2)
            
            # ----------------------------
            # Filter Movies by Rating Range
            # ----------------------------
            st.write("## Filter Movies by Rating")
            try:
                min_rating = float(df[rating_col].min())
                max_rating = float(df[rating_col].max())
            except Exception as e:
                st.error(f"Error determining rating range: {e}")
                return

            if pd.isna(min_rating) or pd.isna(max_rating):
                st.warning("Unable to determine a valid rating range for filtering.")
            else:
                rating_range = st.slider("Select rating range", min_rating, max_rating, (min_rating, max_rating))
                filtered_df = df[(df[rating_col] >= rating_range[0]) & (df[rating_col] <= rating_range[1])]
                st.write(f"Number of movies in the selected range: {filtered_df.shape[0]}")
                st.dataframe(filtered_df.head())
            
            # ----------------------------
            # Regression Analysis Section
            # ----------------------------
            st.write("## Regression Analysis")
            st.write(
                """
                In this section, the IMDb rating is used as the target variable (continuous).
                **Step 1:** Select one or more feature columns from your dataset that can be used to predict the rating.
                **Step 2:** Choose the regression algorithm.
                """
            )
            
            # Multi-select for feature columns (exclude rating)
            available_features = [col for col in df.columns if col not in [rating_col]]
            selected_features = st.multiselect("Select feature columns for regression", options=available_features)
            
            if not selected_features:
                st.error("No feature columns selected for regression. Please select at least one feature column (for example, votes, year).")
            else:
                # Ensure selected features are numeric and drop rows with missing data.
                for col in selected_features:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                before_features_drop = df.shape[0]
                df = df.dropna(subset=selected_features)
                if df.shape[0] < before_features_drop:
                    st.warning("Some rows were dropped due to non-numeric or missing feature values.")
                
                # Verify that there are enough samples for regression.
                if df.shape[0] < 2:
                    st.error("Not enough samples available for regression after filtering. Please adjust feature columns or check your data.")
                else:
                    st.write("### Features used for Regression:")
                    st.write(selected_features)
                    
                    X = df[selected_features]
                    y = df[rating_col]
                    
                    if X.shape[0] < 2:
                        st.error("Not enough samples to perform train/test split. Ensure your dataset contains more samples after filtering.")
                    else:
                        # Split the dataset into training and testing sets.
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )
                        except ValueError as ve:
                            st.error(f"Train/test split error: {ve}")
                            return
                        
                        # ----------------------------
                        # Model Selection for Regression
                        # ----------------------------
                        st.write("### Choose a Regression Algorithm:")
                        model_option = st.radio(
                            "Model Options", 
                            ("Linear Regression", "Decision Tree Regressor", "Random Forest Regressor")
                        )
                        
                        if model_option == "Linear Regression":
                            model = LinearRegression()
                        elif model_option == "Decision Tree Regressor":
                            model = DecisionTreeRegressor(random_state=42)
                        elif model_option == "Random Forest Regressor":
                            model = RandomForestRegressor(random_state=42)
                        else:
                            st.error("Invalid model option selected.")
                            return
                        
                        # ----------------------------
                        # Model Training and Evaluation
                        # ----------------------------
                        try:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                        except Exception as e:
                            st.error(f"Error during model training or prediction: {e}")
                            return
                        
                        try:
                            mae = mean_absolute_error(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                        except Exception as e:
                            st.error(f"Error during evaluation: {e}")
                            return
                        
                        st.markdown(f"""
                        <div style="border: 1px solid #ccc; padding: 10px; background-color: #f7f7f7;">
                        <strong style="font-size:16px;">{'Evaluating ' + model_option}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
                        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
                        st.write(f"**R² Score:** {r2:.2f}")
                        
                        # Optionally, plot the predicted vs. actual ratings.
                        fig3, ax3 = plt.subplots()
                        ax3.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
                        ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                        ax3.set_xlabel("Actual IMDb Rating")
                        ax3.set_ylabel("Predicted IMDb Rating")
                        ax3.set_title("Actual vs. Predicted IMDb Ratings")
                        st.pyplot(fig3)
                        
    else:
        st.info("Upload your IMDb Movie Ratings CSV file to begin.")

if __name__ == '__main__':
    main()
