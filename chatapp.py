from flask import Blueprint, request, jsonify
import os
os.environ['HF_HOME'] = 'D:/huggingface_cache'  # Set FIRST
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline  # Import AFTER

# Create a Blueprint for chatbot routes
chat_bp = Blueprint('chat_bp', __name__)

# Initialize model and template within the Blueprint context
os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 256,
        "repetition_penalty": 1.2,
        "device_map": "auto",
        "torch_dtype": "auto"
    }
)

model = ChatHuggingFace(llm=llm)

template = """<|system|>
You are a specialized Medical Image Diagnostic Assistant. Your purpose is to help users with questions about medical imaging analysis for monkeypox detection, kidney disease analysis, and brain tumor detection. Provide concise, informative responses about using the medical image analysis tools, interpreting results, and general diagnostic guidance. Remember to always clarify that your responses are informational only and not a replacement for professional medical advice.</s>
<|user|>
{input}</s>
<|assistant|>
"""

@chat_bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    
    try:
        formatted_prompt = template.format(input=user_message)
        response = model.invoke(formatted_prompt)
        
        # Remove ALL template tags and special tokens
        cleaned_response = response.content.split("<|assistant|>")[-1]  # Get text after last assistant tag
        cleaned_response = cleaned_response.split("<|endoftext|>")[0]    # Remove end marker
        cleaned_response = cleaned_response.replace("<|system|>", "") \
                                           .replace("<|user|>", "") \
                                           .replace("</s>", "") \
                                           .strip()
        # Remove any remaining whitespace artifacts
        cleaned_response = " ".join(cleaned_response.split())
        
        return jsonify({'response': cleaned_response})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'response': "I'm having trouble processing that. Could you try rephrasing?"})