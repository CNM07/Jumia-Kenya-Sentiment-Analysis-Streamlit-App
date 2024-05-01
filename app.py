import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
from streamlit_option_menu import option_menu
from textblob import TextBlob  
from model import preprocess_data




st.sidebar.image(r"C:\Users\User\Desktop\FT\sent_app\sentiment_analysis_image.png" )
st.sidebar.title("About")
about_text = """
    Welcome!
    This app conducts sentiment analysis and  of mobile phone devices and accessories reviews on Jumia Kenya. Powered by natural language processing techniques and random forest modeling, the app uncovers sentiments and delivers valuable insights from the vast landscape of customer feedback. Simplify your decision-making process and unlock the power of customer sentiment with this intuitive app.
    """
st.sidebar.markdown(f'<div style="background-color: #d3d3d3; padding: 10px; border-radius: 5px;">{about_text}</div>', unsafe_allow_html=True)


# Load the pre-trained Random Forest model
with open("random_forest_model.pkl", "rb") as f:
    random_forest_model = pickle.load(f)

# Load the TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Main function for Streamlit app
def main():
    st.title("Jumia Kenya Sentiment Analysis App")
    #st.subheader(" Sentiment Analysis on Mobile Phone Devices and Smartphone Accessories Sold on Jumia Kenya")
    st.write("Sentiment Analysis on Mobile Phone Devices and Smartphone Accessories Sold on Jumia Kenya")

# Text input
    
    with st.form(key="nlpForm"):
            text_input = st.text_area("Enter text here:")
            submit_button = st.form_submit_button(label="Analyze Sentiment")

    st.markdown("---")


    # Button to perform sentiment analysis
    
    if submit_button:
            processed_text = preprocess_data(pd.Series(text_input))
            
        # Display original and processed text side by side in a table
            table_data = {
                "Original Text": [text_input],
                "Pre-Processed Text": [processed_text[0]]
            }
            st.table(pd.DataFrame(table_data))

            st.markdown("---")

            col1, col2 = st.columns(2)    

            with col1:
                # Transform input text using TF-IDF vectorizer
                X_input = vectorizer.transform(processed_text)

                blob = TextBlob(text_input)
                polarity = blob.sentiment.polarity
                
                if polarity > 0:
                    sentiment_class = 'Positive'
                    emoji = '😊'
                elif polarity < 0:
                    sentiment_class = 'Negative'
                    emoji = '😠'
                else:
                    sentiment_class = 'Neutral'
                    emoji = '😐'

                #st.write("**Polarity:**", polarity)
                
                st.write(f'**Results:** The given review is {sentiment_class} {emoji}')


                # RF Predict sentiment
                prediction = random_forest_model.predict(X_input)[0]

                # Get predicted sentiment class
                sentiment_classes = ['Negative', 'Neutral', 'Positive']
                predicted_class = sentiment_classes[prediction]
                #predicted_class = prediction

                # Display predicted sentiment
                #st.write(f"**Predicted Class:** {predicted_class}")
                #st.write(f"Predicted: {predicted_class}")


            with col2:
                if sentiment_class == 'Positive':
                    #st.markdown("Sentiment: Positive 😊")
                    st.image(r"C:\Users\User\Desktop\FT\sent_app\positive.jpg", width=100)

                elif sentiment_class == 'Neutral':
                    #st.markdown("Sentiment: Neutral 😐")
                    st.image(r"C:\Users\User\Desktop\FT\sent_app\neutral.png", width=100)

                else:
                    #st.markdown("Sentiment: Negative 😠")
                    st.image(r"C:\Users\User\Desktop\FT\sent_app\negative.png", width=100)

                 
    
        
        
    
    


        




if __name__ == "__main__":
    main()