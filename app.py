import os
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# --- CONFIGURATION ---
# Your API Key is embedded here as requested
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key

# 1. Initialize the modern GenAI Client
client = genai.Client(
    api_key=API_KEY, 
    http_options={'api_version': 'v1beta'}
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_blog_post', methods=['POST'])
def generate_blog_post():
    try:
        data = request.get_json()
        topic = data.get('topic')
        keywords = data.get('keywords')
        tone = data.get('tone', 'professional')
        length = data.get('length', 'medium')

        if not topic:
            return jsonify({"error": "Topic is required"}), 400

        # Constructing the prompt
        prompt = f"Write a {length} blog post about: '{topic}'.\n"
        if keywords:
            prompt += f"Keywords to include: {keywords}.\n"
        if tone:
            prompt += f"Maintain a {tone} tone throughout.\n"
        
        # Mapping length to token counts
        max_output_tokens = {
            "short": 400,
            "medium": 1000,
            "long": 2048
        }.get(length, 1000)

        # 2. Generate content using the client object
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful AI assistant that generates engaging, SEO-friendly blog posts.",
                max_output_tokens=max_output_tokens,
                temperature=0.7,
            )
        )

        # 3. Extract the text
        blog_content = response.text
        
        return jsonify({"blog_post": blog_content})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)