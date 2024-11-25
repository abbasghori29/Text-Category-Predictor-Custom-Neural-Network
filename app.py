import streamlit as st
import pickle
import numpy as np
import re
from PIL import Image
import plotly.graph_objects as go
import time
import pandas as pd

import pickle
import numpy as np
import re

class TP:
    def __init__(self, model_path, vectorizer_path):
        """
        Initialize the predictor with paths to the saved model and vectorizer.
        
        Args:
            model_path (str): Path to the saved model pickle file
            vectorizer_path (str): Path to the saved vectorizer pickle file
        """
        self.class_names = ['World', 'Sports', 'Business', 'Science']
        self.model = None
        self.vectorizer = None
        self.load_artifacts(model_path, vectorizer_path)
        
    def load_artifacts(self, model_path, vectorizer_path):
        """Load the saved model and vectorizer."""
        try:
            # Load the vectorizer
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
                
            # Load the model parameters
            with open(model_path, 'rb') as f:
                model_params = pickle.load(f)
                
            # Recreate the neural network with loaded parameters
            input_size = model_params['weights1'].shape[0]
            hidden_size = model_params['weights1'].shape[1]
            output_size = model_params['weights2'].shape[1]
            
            self.model = NeuralNetwork(input_size, hidden_size, output_size)
            self.model.set_parameters(model_params)
            
            print("Model and vectorizer loaded successfully!")
            
        except Exception as e:
            raise Exception(f"Error loading model artifacts: {str(e)}")
    
    def preprocess_text(self, text):
        """Preprocess the input text."""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        return text
    
    def predict(self, text):
        """
        Predict the category of the input text.
        
        Args:
            text (str): Input text to classify
            
        Returns:
            dict: Prediction results containing category and confidence scores
        """
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(text)
            
            # Transform text using the vectorizer
            text_vectorized = self.vectorizer.transform([processed_text]).toarray()
            
            # Get model predictions
            predictions = self.model.forward(text_vectorized)
            
            # Get the predicted class and probabilities
            predicted_class_idx = np.argmax(predictions[0])
            probabilities = predictions[0]
            
            # Create results dictionary
            results = {
                'predicted_category': self.class_names[predicted_class_idx],
                'confidence': float(probabilities[predicted_class_idx]),
                'probabilities': {
                    category: float(prob) 
                    for category, prob in zip(self.class_names, probabilities)
                }
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")

class NeuralNetwork:
    """Simplified version of the neural network for prediction only."""
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.initialize_weights()

    def initialize_weights(self):
        self.weights1 = np.zeros((self.input_size, self.hidden_size))
        self.weights2 = np.zeros((self.hidden_size, self.output_size))
        self.bias1 = np.zeros((1, self.hidden_size))
        self.bias2 = np.zeros((1, self.output_size))

    def set_parameters(self, parameters):
        self.weights1 = parameters['weights1'].copy()
        self.weights2 = parameters['weights2'].copy()
        self.bias1 = parameters['bias1'].copy()
        self.bias2 = parameters['bias2'].copy()

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        layer1 = np.dot(X, self.weights1) + self.bias1
        layer1_activation = self.relu(layer1)
        layer2 = np.dot(layer1_activation, self.weights2) + self.bias2
        output = self.softmax(layer2)
        return output# 1. First, set up your file paths correctly 
model_path = "model_artifacts_20241125_082544/best_model.pkl"    # Replace with your model file path
vectorizer_path = "tfidf_vectorizer.pkl"  # Replace with your vectorizer file path

# 2. Create the predictor
predictor = TP(model_path, vectorizer_path)

# 3. Input your sentence and get prediction
sentence = """
While the synergy between our vertically integrated strategies continues to amplify market penetration, 
the Q4 projections remain contingent on the elasticity of discretionary consumer spending amidst fluctuating interest rate policies.
"""  # Put your sentence here
result = predictor.predict(sentence)

# 4. See the results
print(f"\nText: {sentence}")
print(f"Predicted Category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.2%}")
print("\nProbabilities for all categories:")
for category, prob in result['probabilities'].items():
    print(f"{category}: {prob:.2%}")

def create_gauge_chart(probability, category):
    """Create a gauge chart for probability visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': category},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(height=200)
    return fig

def main():
    st.set_page_config(page_title="Text Category Predictor", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        .stTextInput > div > div > input {
            font-size: 18px;
            padding: 15px;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin: 10px 0;
        }
        .big-font {
            font-size: 24px;
            font-weight: bold;
            color: #0e1117;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("📝 Text Category Predictor")
    st.markdown("Enter your text below to classify it into one of the following categories: World, Sports, Business, or Science")
    
    # Load model and vectorizer
    @st.cache_resource
    def load_model():
        predictor = TP(model_path, vectorizer_path)
        return predictor
    
    try:
        predictor = load_model()
        
        # Text input
        text_input = st.text_area("Enter your text here:", height=150,
                                 placeholder="Type or paste your text here...")
        
        if st.button("Predict Category", type="primary"):
            if text_input.strip():
                # Show spinner during prediction
                with st.spinner('Analyzing text...'):
                    time.sleep(1)  # Add small delay for better UX
                    result = predictor.predict(text_input)
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### 🎯 Prediction Results")
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <p class="big-font">Predicted Category: {result['predicted_category']}</p>
                            <p>Confidence: {result['confidence']:.2%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                    # Create gauge charts for all categories
                    st.markdown("### 📊 Confidence Scores by Category")
                    st.pyplot(pd.DataFrame(result['probabilities'].values(), index=result['probabilities'].keys(), columns=['Probability']).plot(kind='barh', figsize=(5, 3), color='skyblue', legend=False, title='Prediction Probabilities').get_figure())
                
                with col2:
                    # Word count and characteristics
                    st.markdown("### 📝 Text Statistics")
                    words = len(text_input.split())
                    chars = len(text_input)
                    sentences = len(text_input.split('.'))
                    
                    stats_html = f"""
                    <div class="prediction-box">
                        <p>Word Count: {words}</p>
                        <p>Character Count: {chars}</p>
                        <p>Sentence Count: {sentences}</p>
                    </div>
                    """
                    st.markdown(stats_html, unsafe_allow_html=True)
            else:
                st.warning("Please enter some text before predicting.")
        
        # Add helpful instructions at the bottom
        with st.expander("ℹ️ How to use this app"):
            st.markdown("""
                1. Enter or paste your text in the text area above
                2. Click the 'Predict Category' button
                3. View the predicted category and confidence scores
                4. The gauge charts show the probability for each category
                5. Text statistics provide additional insights about your input
            """)
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.markdown("Please make sure the model and vectorizer files are in the correct location.")

if __name__ == "__main__":
    main()