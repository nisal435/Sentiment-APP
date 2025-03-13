import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
from collections import Counter

# FastAPI URL (Backend)
API_URL = "http://127.0.0.1:8000/predict/"  # Change this when deploying

# Initialize an empty list or DataFrame to store sentiment results in session state
if 'sentiment_data' not in st.session_state:
    st.session_state.sentiment_data = []

# Custom CSS to add the image in the top-right corner as a watermark
st.markdown("""
    <style>
        .reportview-container {
            position: relative;
        }
        .watermark {
            position: absolute;
            top: 0;
            right: 0;
            width: 200px;  /* Adjust the size of the image */
            opacity: 0.9;  /* Adjust the opacity for watermark effect */
            z-index: 1;  /* Ensure the watermark is above content */
        }
    </style>
""", unsafe_allow_html=True)

# Add the image to the top-right corner as a watermark
st.markdown("""
    <div class="watermark">
        <img src="https://thumbs.dreamstime.com/b/male-chef-preparing-dish-food-review-website-page-flat-vector-illustration-symbolizing-online-criticism-culinary-art-342153061.jpg" />
    </div>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("🍽️ TasteMood Analyzer")
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Find out the mood behind your words!")

st.markdown("<br>", unsafe_allow_html=True)
# Text input from user
user_input = st.text_area("Tell us about your food experience:")

st.markdown("<br>", unsafe_allow_html=True)
# If button is clicked
if st.button("Analyze Sentiment"):
    if user_input:
        try:
            # Show loading spinner while waiting for response
            with st.spinner("Analyzing sentiment..."):
                # Make a POST request to the FastAPI backend
                response = requests.post(API_URL, json={"text": user_input})

            # If the response is successful
            if response.status_code == 200:
                result = response.json()

                # Get the sentiment label and score
                sentiment_label = result['label']
                confidence_score = result['score']

                # Add sentiment data to session state
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                sentiment_data = {
                    "text": user_input, 
                    "label": sentiment_label, 
                    "score": confidence_score, 
                    "timestamp": timestamp
                }
                st.session_state.sentiment_data.append(sentiment_data)

                # Display the result
                st.write(f"**Sentiment:** {sentiment_label}")
                st.write(f"**Confidence Score:** {confidence_score:.4f}")
                
            else:
                # Show an error if backend didn't respond correctly
                st.error(f"Error processing request: {response.status_code}")
        except requests.exceptions.RequestException as e:
            # Show a different error if there is a network issue
            st.error(f"Error connecting to the backend: {e}")
    else:
        # If no input provided
        st.warning("Please enter some text first!")

# Display the data collected in a table if there are any sentiment results
st.markdown("<br>", unsafe_allow_html=True)
if st.session_state.sentiment_data:
    st.subheader("Sentiment Analysis History")
    sentiment_df = pd.DataFrame(st.session_state.sentiment_data)

    # Format the timestamp to be more readable
    sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
    st.write(sentiment_df)

  # Plot a trend of sentiment scores over time with red line color
    fig = px.line(
    sentiment_df, 
    x='timestamp', 
    y='score', 
    title='Sentiment Score Trend Over Time', 
    markers=True
    )

    # Update layout and set line color to red
    fig.update_traces(line=dict(color='red'))  # Set line color to red
    fig.update_layout(xaxis_title='Timestamp', yaxis_title='Confidence Score')

    # Display the chart
    st.plotly_chart(fig)

    # **Second Chart**: Visualize sentiment distribution (positive, negative, neutral) with custom colors
    sentiment_counts = Counter([data['label'] for data in st.session_state.sentiment_data])
    sentiment_labels = list(sentiment_counts.keys())
    sentiment_values = list(sentiment_counts.values())

    fig = px.pie(
        names=sentiment_labels,
        values=sentiment_values,
        title="Sentiment Distribution",
        labels={'names': 'Sentiment', 'values': 'Percentage'},
        color=sentiment_labels,  # Map colors based on sentiment labels
        color_discrete_map={
            'Positive': '#6b110e',  # Positive sentiment in #28A745 (Green)
            'Negative': '#a32622',  # Negative sentiment in #D9534F (Muted Red)
        }
    )
    st.plotly_chart(fig)

else:
    st.write("No sentiment data available yet.") 
