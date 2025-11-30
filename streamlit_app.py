"""
Streamlit Interface for Churn Prediction API
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any
import json

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .risk-low {
        color: #00cc00;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if API is running and healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("models_loaded", False)
        return False
    except Exception:
        return False


def predict_single_customer(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    """Send single customer data to API for prediction"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=customer_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}


def predict_batch_customers(customers_data: list) -> Dict[str, Any]:
    """Send batch customer data to API for prediction"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json={"customers": customers_data},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}


def display_prediction_result(result: Dict[str, Any]):
    """Display prediction result with visualizations"""
    
    churn = result["churn_prediction"]
    probability = result["churn_probability"]
    confidence = result["confidence"]
    risk_level = result["risk_level"]
    
    # Columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Churn Prediction",
            value="YES" if churn == 1 else "NO",
            delta="Will Churn" if churn == 1 else "Will Stay"
        )
    
    with col2:
        st.metric(
            label="Churn Probability",
            value=f"{probability:.2%}"
        )
    
    with col3:
        st.metric(
            label="Confidence",
            value=f"{confidence:.2%}"
        )
    
    with col4:
        risk_class = "risk-high" if risk_level == "High" else "risk-medium" if risk_level == "Medium" else "risk-low"
        st.markdown(f"**Risk Level**")
        st.markdown(f"<p class='{risk_class}'>{risk_level}</p>", unsafe_allow_html=True)
    
    # Probability gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#90EE90'},
                {'range': [30, 60], 'color': '#FFD700'},
                {'range': [60, 100], 'color': '#FF6B6B'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def display_batch_results(results: Dict[str, Any]):
    """Display batch prediction results"""
    
    predictions = results["predictions"]
    total_customers = results["total_customers"]
    churn_count = results["churn_count"]
    churn_percentage = results["churn_percentage"]
    
    # Summary metrics
    st.subheader("📊 Batch Analysis Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Customers", total_customers)
    with col2:
        st.metric("Predicted Churners", churn_count)
    with col3:
        st.metric("Churn Rate", f"{churn_percentage:.2f}%")
    
    # Create DataFrame for visualization
    df = pd.DataFrame([
        {
            "Customer": i + 1,
            "Churn": "Yes" if p["churn_prediction"] == 1 else "No",
            "Probability": p["churn_probability"],
            "Confidence": p["confidence"],
            "Risk Level": p["risk_level"]
        }
        for i, p in enumerate(predictions)
    ])
    
    # Risk distribution pie chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Distribution")
        churn_dist = df["Churn"].value_counts()
        fig_pie = px.pie(
            values=churn_dist.values,
            names=churn_dist.index,
            color=churn_dist.index,
            color_discrete_map={"Yes": "#ff4b4b", "No": "#00cc00"}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Risk Level Distribution")
        risk_dist = df["Risk Level"].value_counts()
        fig_risk = px.bar(
            x=risk_dist.index,
            y=risk_dist.values,
            color=risk_dist.index,
            color_discrete_map={"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#00cc00"},
            labels={"x": "Risk Level", "y": "Count"}
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Probability distribution
    st.subheader("Churn Probability Distribution")
    fig_hist = px.histogram(
        df,
        x="Probability",
        nbins=20,
        color="Churn",
        color_discrete_map={"Yes": "#ff4b4b", "No": "#00cc00"}
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Detailed results table
    st.subheader("Detailed Results")
    st.dataframe(
        df.style.background_gradient(subset=["Probability"], cmap="RdYlGn_r"),
        use_container_width=True
    )


def main():
    """Main application"""
    
    # Header
    st.markdown('<p class="main-header">📞 Customer Churn Prediction System</p>', unsafe_allow_html=True)
    
    # Check API health
    with st.sidebar:
        st.header("🔌 API Status")
        if check_api_health():
            st.success("✅ API Connected")
            st.success("✅ Models Loaded")
        else:
            st.error("❌ API Not Available")
            st.warning("Please ensure the API server is running at http://localhost:8000")
            st.code("python app.py", language="bash")
            st.stop()
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.info("""
        This application predicts customer churn using machine learning.
        
        **Features:**
        - Single customer prediction
        - Batch predictions from CSV
        - Risk level assessment
        - Interactive visualizations
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📁 Batch Prediction", "📖 User Guide"])
    
    with tab1:
        st.header("Single Customer Prediction")
        st.write("Enter customer information to predict churn probability")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Information")
            state = st.text_input("State", value="NY", max_chars=2)
            account_length = st.number_input("Account Length (days)", min_value=1, value=128)
            area_code = st.selectbox("Area Code", [415, 510, 408])
            international_plan = st.selectbox("International Plan", ["No", "Yes"])
            voice_mail_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])
            number_vmail_messages = st.number_input("Number of Voicemail Messages", min_value=0, value=25)
            customer_service_calls = st.number_input("Customer Service Calls", min_value=0, value=1)
        
        with col2:
            st.subheader("Usage Statistics")
            
            st.write("**Day Time**")
            total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, value=265.1)
            total_day_calls = st.number_input("Total Day Calls", min_value=0, value=110)
            total_day_charge = st.number_input("Total Day Charge", min_value=0.0, value=45.07)
            
            st.write("**Evening Time**")
            total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, value=197.4)
            total_eve_calls = st.number_input("Total Evening Calls", min_value=0, value=99)
            total_eve_charge = st.number_input("Total Evening Charge", min_value=0.0, value=16.78)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("**Night Time**")
            total_night_minutes = st.number_input("Total Night Minutes", min_value=0.0, value=244.7)
            total_night_calls = st.number_input("Total Night Calls", min_value=0, value=91)
            total_night_charge = st.number_input("Total Night Charge", min_value=0.0, value=11.01)
        
        with col4:
            st.write("**International**")
            total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0, value=10.0)
            total_intl_calls = st.number_input("Total International Calls", min_value=0, value=3)
            total_intl_charge = st.number_input("Total International Charge", min_value=0.0, value=2.7)
        
        if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
            customer_data = {
                "State": state,
                "Account length": account_length,
                "Area code": area_code,
                "International plan": international_plan,
                "Voice mail plan": voice_mail_plan,
                "Number vmail messages": number_vmail_messages,
                "Total day minutes": total_day_minutes,
                "Total day calls": total_day_calls,
                "Total day charge": total_day_charge,
                "Total eve minutes": total_eve_minutes,
                "Total eve calls": total_eve_calls,
                "Total eve charge": total_eve_charge,
                "Total night minutes": total_night_minutes,
                "Total night calls": total_night_calls,
                "Total night charge": total_night_charge,
                "Total intl minutes": total_intl_minutes,
                "Total intl calls": total_intl_calls,
                "Total intl charge": total_intl_charge,
                "Customer service calls": customer_service_calls
            }
            
            with st.spinner("Making prediction..."):
                result = predict_single_customer(customer_data)
            
            if result["success"]:
                st.success("✅ Prediction completed!")
                st.markdown("---")
                display_prediction_result(result["data"])
            else:
                st.error(f"❌ Prediction failed: {result['error']}")
    
    with tab2:
        st.header("Batch Prediction from CSV")
        st.write("Upload a CSV file with customer data for batch predictions")
        
        # Sample CSV download
        sample_data = pd.DataFrame([{
            "State": "NY",
            "Account length": 128,
            "Area code": 415,
            "International plan": "No",
            "Voice mail plan": "Yes",
            "Number vmail messages": 25,
            "Total day minutes": 265.1,
            "Total day calls": 110,
            "Total day charge": 45.07,
            "Total eve minutes": 197.4,
            "Total eve calls": 99,
            "Total eve charge": 16.78,
            "Total night minutes": 244.7,
            "Total night calls": 91,
            "Total night charge": 11.01,
            "Total intl minutes": 10.0,
            "Total intl calls": 3,
            "Total intl charge": 2.7,
            "Customer service calls": 1
        }])
        
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=sample_data.to_csv(index=False),
            file_name="sample_customers.csv",
            mime="text/csv"
        )
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.subheader("Preview of uploaded data")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                    customers_data = df.to_dict('records')
                    
                    with st.spinner(f"Making predictions for {len(customers_data)} customers..."):
                        result = predict_batch_customers(customers_data)
                    
                    if result["success"]:
                        st.success("✅ Batch prediction completed!")
                        st.markdown("---")
                        display_batch_results(result["data"])
                        
                        # Download results
                        results_df = pd.DataFrame([
                            {
                                "Customer": i + 1,
                                "Churn": p["churn_prediction"],
                                "Probability": p["churn_probability"],
                                "Confidence": p["confidence"],
                                "Risk Level": p["risk_level"]
                            }
                            for i, p in enumerate(result["data"]["predictions"])
                        ])
                        
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=results_df.to_csv(index=False),
                            file_name="churn_predictions.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"❌ Batch prediction failed: {result['error']}")
                        
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")
    
    with tab3:
        st.header("📖 User Guide")
        
        st.subheader("How to Use This Application")
        
        st.markdown("""
        ### Single Prediction
        1. Navigate to the **Single Prediction** tab
        2. Fill in all customer information fields
        3. Click the **Predict Churn** button
        4. View the prediction results and visualizations
        
        ### Batch Prediction
        1. Navigate to the **Batch Prediction** tab
        2. Download the sample CSV template
        3. Fill in your customer data following the template format
        4. Upload your CSV file
        5. Click **Run Batch Prediction**
        6. View the comprehensive analysis and download results
        
        ### Understanding Results
        
        **Churn Prediction**: YES/NO - Whether the customer is likely to churn
        
        **Churn Probability**: 0-100% - Likelihood of customer churning
        
        **Confidence**: How confident the model is in its prediction
        
        **Risk Levels**:
        - 🟢 **Low** (0-30%): Customer is unlikely to churn
        - 🟡 **Medium** (30-60%): Customer shows moderate churn risk
        - 🔴 **High** (60-100%): Customer has high churn risk
        
        ### Required Data Fields
        - **State**: Two-letter state code
        - **Account length**: Number of days customer has been with company
        - **Area code**: Customer's area code (415, 510, or 408)
        - **International plan**: Yes/No
        - **Voice mail plan**: Yes/No
        - **Usage statistics**: Minutes, calls, and charges for day/evening/night/international
        - **Customer service calls**: Number of calls to customer service
        """)
        
        st.subheader("API Information")
        st.code(f"API Base URL: {API_BASE_URL}", language="text")
        
        st.markdown("""
        ### Troubleshooting
        
        **API Not Available**:
        - Ensure the FastAPI server is running
        - Run: `python app.py` in your terminal
        - Check that it's running on http://localhost:8000
        
        **CSV Upload Issues**:
        - Ensure column names match the template exactly
        - Check for any missing values
        - Verify data types are correct
        """)


if __name__ == "__main__":
    main()