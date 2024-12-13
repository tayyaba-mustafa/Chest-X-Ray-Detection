import streamlit as st
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from torchvision import transforms
from PIL import Image
from groq import Groq
from fpdf import FPDF
import os

class MedicalDiagnosisApp:
    def __init__(self):
        # Team Name
        self.team_name = "MedAI Diagnostics"
        
        # Define class mapping
        self.class_mapping = {
            0: "Atelectasis",
            1: "Cardiomegaly",
            2: "Effusion",
            3: "Infiltration",
            4: "Mass",
            5: "Nodule",
            6: "Pneumonia",
            7: "Pneumothorax",
            8: "Consolidation",
            9: "Edema",
            10: "Emphysema",
            11: "Fibrosis",
            12: "Pleural Thickening",
            13: "Hernia",
        }
        
        # Initialize model and transforms with enhanced preprocessing
        self.model = self._load_pretrained_model()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize Groq client using Streamlit secrets
        self.groq_client = self.initialize_groq_client()
    
    def _load_pretrained_model(self):
        """Load a more robust pre-trained model"""
        try:
            model = timm.create_model('densenet121', pretrained=True, num_classes=14)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    def initialize_groq_client(self):
        """Initialize Groq client using Streamlit secrets"""
        try:
            # Try to get API key from Streamlit secrets
            api_key = st.secrets.get("GROQ_API_KEY")
            
            if not api_key:
                st.warning("Groq API key not found. Some AI features will be disabled.")
                return None
            
            return Groq(api_key=api_key)
        except Exception as e:
            st.error(f"Error initializing Groq client: {e}")
            return None
    
    def generate_disease_description(self, disease_name):
        """Generate detailed disease description using Groq"""
        if not self.groq_client:
            return "AI description service is currently unavailable. Please check your API configuration."
        
        try:
            # Prompt for comprehensive medical description
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical expert providing detailed, accurate medical information."
                    },
                    {
                        "role": "user",
                        "content": f"""Provide a comprehensive medical description for {disease_name}. 
                        Include:
                        1. Medical definition
                        2. Common symptoms
                        3. Potential causes
                        4. Diagnostic methods
                        5. Treatment options
                        6. Preventive measures
                        
                        Ensure the information is clear, scientific, and patient-friendly."""
                    }
                ],
                model="llama3-8b-8192",
                max_tokens=700
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error generating description: {str(e)}"
    
    def medical_chatbot(self, user_input):
        """Medical chatbot using Groq"""
        if not self.groq_client:
            return "Chatbot service is currently unavailable. Please check your API configuration."
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful medical AI assistant providing general health information. Always emphasize consulting a healthcare professional for personalized advice."
                    },
                    {
                        "role": "user",
                        "content": user_input
                    }
                ],
                model="llama3-8b-8192",
                max_tokens=500
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Chatbot error: {str(e)}"
    
    def preprocess_image(self, image):
        """Enhanced image preprocessing"""
        try:
            # Convert to grayscale and then to RGB to handle different image formats
            image_gray = image.convert('L')
            image_rgb = image_gray.convert('RGB')
            return self.transform(image_rgb).unsqueeze(0)
        except Exception as e:
            st.error(f"Image preprocessing error: {e}")
            return None
    
    def predict_disease(self, input_tensor):
        """Advanced prediction with confidence scores"""
        if input_tensor is None or self.model is None:
            st.error("Model or input is not properly initialized")
            return None
        
        try:
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_k_values, top_k_indices = torch.topk(probabilities, k=3)
            
            # Convert to numpy for easier handling
            top_k_values = top_k_values.numpy()[0]
            top_k_indices = top_k_indices.numpy()[0]
            
            # Create results with diseases and their probabilities
            results = [
                {
                    'disease': self.class_mapping.get(idx, 'Unknown'),
                    'probability': prob * 100
                }
                for idx, prob in zip(top_k_indices, top_k_values)
            ]
            
            return results
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None

def plot_disease_probabilities(results):
    """Create a bar plot of disease probabilities"""
    try:
        plt.figure(figsize=(10, 6))
        diseases = [result['disease'] for result in results]
        probabilities = [result['probability'] for result in results]
        
        sns.barplot(x=probabilities, y=diseases, palette='viridis')
        plt.title('Top Predicted Chest X-Ray Conditions', fontsize=15)
        plt.xlabel('Probability (%)', fontsize=12)
        plt.ylabel('Conditions', fontsize=12)
        plt.tight_layout()
        
        # Save the plot to a buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Plotting error: {e}")
        return None

# Main function to run the Streamlit app
def main():
    # Configure page first, before any other Streamlit commands
    st.set_page_config(
        page_title="MedAI Diagnostics - Medical Diagnosis",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize the app
    app = MedicalDiagnosisApp()
    
    # Sidebar for navigation
    st.sidebar.title("🏥 LungLens")
    
    # Sidebar menu
    nav_option = st.sidebar.radio(
        "Navigate", 
        [
            "X-Ray Diagnosis", 
            "Medical Chatbot", 
            "About Us"
        ]
    )
    
    # Conditional page rendering
    if nav_option == "X-Ray Diagnosis":
        xray_diagnosis_page(app)
    elif nav_option == "Medical Chatbot":
        chatbot_page(app)
    else:
        about_page(app)

def xray_diagnosis_page(app):
    """X-Ray Diagnosis Page"""
    st.markdown("# 🩺 Medical Image Diagnosis")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Chest X-Ray", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Open and process image
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)

        with col2:
            # Preprocess and predict
            input_tensor = app.preprocess_image(image)
            prediction_results = app.predict_disease(input_tensor)

            if prediction_results:
                # Display top predictions
                st.markdown("## Prediction Results")
                for result in prediction_results:
                    st.metric(
                        label=result['disease'], 
                        value=f"{result['probability']:.2f}%"
                    )

                # Plot probabilities
                try:
                    plot_buffer = plot_disease_probabilities(prediction_results)
                    if plot_buffer:
                        st.image(plot_buffer, caption="Probability Distribution", use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate probability plot: {e}")

                # Generate and display AI-powered description
                with st.spinner("Generating detailed medical description..."):
                    # Use the top predicted disease
                    top_disease = prediction_results[0]['disease']
                    disease_description = app.generate_disease_description(top_disease)

                    # Expandable description
                    with st.expander("Detailed Medical Description"):
                        st.write(disease_description)

                



def chatbot_page(app):
    """Medical Chatbot Interface"""
    st.markdown("# 💬 Medical AI Chatbot")
    
    # Chat input
    user_input = st.text_input("Ask a medical question:")
    
    if user_input:
        # Generate AI response
        with st.spinner("Generating response..."):
            response = app.medical_chatbot(user_input)
            
            # Display response
            st.markdown("### AI Response")
            st.info(response)
            
            # Disclaimer
            st.warning("Note: This is AI-generated information. Always consult a healthcare professional for personalized medical advice.")

def about_page(app):
    """About the team page"""
    st.markdown("# 🌟 About LungLens AI")
    st.markdown("""
    ## Our Mission
    Leveraging AI to provide intelligent medical insights and support.
    
    ### Our Approach
    - Advanced Machine Learning
    - Comprehensive Medical Information
    - Patient-Centric Design
    
    **Disclaimer:** Our AI tools are meant to assist, not replace, professional medical consultation.
    """)

# Run the Streamlit app
if __name__ == "__main__":
    main()
