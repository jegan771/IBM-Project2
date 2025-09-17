# IBM-Project2
[Watch demo video](https://drive.google.com/file/d/1pOBbB92T1syjO9p4hHT2olIcC53ikCAz/view?usp=drivesdk)




📘 Project Documentation: City Analysis & Citizen Services AI

1. Overview

The City Analysis & Citizen Services AI is an interactive web application powered by the IBM Granite 3.2-2B Instruct language model.
It provides two key functionalities:

1. City Analysis – Generates detailed reports about cities, covering crime statistics, accident rates, and overall safety.


2. Citizen Services Assistant – Responds to public queries about government services, policies, and civic issues.



The app is built using Gradio for the frontend and Hugging Face Transformers for model inference.


---

2. Features

🔹 City Analysis

Input: City name (e.g., New York, London, Mumbai)

Output:

Crime Index & Safety statistics

Accident & traffic safety information

Overall safety assessment



🔹 Citizen Services

Input: Citizen query related to public services or policies

Output: AI-generated government-style response



---

3. Tech Stack

Python – Core programming language

Gradio – Web UI framework for interaction

Transformers (Hugging Face) – Model inference pipeline

PyTorch – Backend for model execution

IBM Granite 3.2-2B Instruct – Pretrained LLM for natural language generation



---

4. Installation & Setup

📦 Requirements

pip install gradio torch transformers

▶️ Running the App

Save the script as app.py and run:

python app.py

The app will launch locally at:

http://127.0.0.1:7860

If share=True, it generates a public link for external access.


---

5. Code Walkthrough

🔹 Model Loading

model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

Loads tokenizer and model from Hugging Face Hub

Uses GPU (float16) if available, else CPU (float32)



---

🔹 Text Generation

def generate_response(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

Handles text-to-text generation

Uses sampling (do_sample=True) with temperature=0.7 for natural responses



---

🔹 City Analysis Function

def city_analysis(city_name):
    prompt = f"Provide a detailed analysis of {city_name} including:\n1. Crime Index and safety statistics\n2. Accident rates and traffic safety information\n3. Overall safety assessment\n\nCity: {city_name}\nAnalysis:"
    return generate_response(prompt, max_length=1000)


---

🔹 Citizen Interaction Function

def citizen_interaction(query):
    prompt = f"As a government assistant, provide accurate and helpful information about the following citizen query related to public services, government policies, or civic issues:\n\nQuery: {query}\nResponse:"
    return generate_response(prompt, max_length=1000)


---

🔹 Gradio UI

Tab 1 – City Analysis → Input city name → Output analysis

Tab 2 – Citizen Services → Input query → Output response


with gr.Blocks() as app:
    gr.Markdown("# City Analysis & Citizen Services AI")
    ...
app.launch(share=True)


---

6. Example Usage

Input (City Analysis):

Mumbai

Output:

Crime Index: Moderate  
Traffic Safety: High accident risk in peak hours  
Overall Safety: Requires caution in crowded areas, better in residential zones.


---

Input (Citizen Services):

What are the eligibility criteria for free public healthcare?

Output:

Eligibility is determined based on income levels, residency status, and government ID proof. Low-income families and senior citizens receive priority benefits.


---

7. Future Improvements

Add real-time data integration (crime/accident stats from APIs)

Enhance multi-language support

Store query history & allow PDF export of reports

Deploy on cloud platforms (AWS, GCP, Hugging Face Spaces)



---

✅ That’s a full documentation draft.
Do you want me to also create a README.md version (GitHub-friendly) or a PDF project report version?

Got it 👍 You want project documentation for the Gradio app you built with IBM Granite 3.2 instruct model. Here’s a structured version you can use:


---

📘 Project Documentation: City Analysis & Citizen Services AI

1. Overview

The City Analysis & Citizen Services AI is an interactive web application powered by the IBM Granite 3.2-2B Instruct language model.
It provides two key functionalities:

1. City Analysis – Generates detailed reports about cities, covering crime statistics, accident rates, and overall safety.


2. Citizen Services Assistant – Responds to public queries about government services, policies, and civic issues.



The app is built using Gradio for the frontend and Hugging Face Transformers for model inference.


---

2. Features

🔹 City Analysis

Input: City name (e.g., New York, London, Mumbai)

Output:

Crime Index & Safety statistics

Accident & traffic safety information

Overall safety assessment



🔹 Citizen Services

Input: Citizen query related to public services or policies

Output: AI-generated government-style response



---

3. Tech Stack

Python – Core programming language

Gradio – Web UI framework for interaction

Transformers (Hugging Face) – Model inference pipeline

PyTorch – Backend for model execution

IBM Granite 3.2-2B Instruct – Pretrained LLM for natural language generation



---

4. Installation & Setup

📦 Requirements

pip install gradio torch transformers

▶️ Running the App

Save the script as app.py and run:

python app.py

The app will launch locally at:

http://127.0.0.1:7860

If share=True, it generates a public link for external access.


---

5. Code Walkthrough

🔹 Model Loading

model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

Loads tokenizer and model from Hugging Face Hub

Uses GPU (float16) if available, else CPU (float32)



---

🔹 Text Generation

def generate_response(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

Handles text-to-text generation

Uses sampling (do_sample=True) with temperature=0.7 for natural responses



---

🔹 City Analysis Function

def city_analysis(city_name):
    prompt = f"Provide a detailed analysis of {city_name} including:\n1. Crime Index and safety statistics\n2. Accident rates and traffic safety information\n3. Overall safety assessment\n\nCity: {city_name}\nAnalysis:"
    return generate_response(prompt, max_length=1000)


---

🔹 Citizen Interaction Function

def citizen_interaction(query):
    prompt = f"As a government assistant, provide accurate and helpful information about the following citizen query related to public services, government policies, or civic issues:\n\nQuery: {query}\nResponse:"
    return generate_response(prompt, max_length=1000)


---

🔹 Gradio UI

Tab 1 – City Analysis → Input city name → Output analysis

Tab 2 – Citizen Services → Input query → Output response


with gr.Blocks() as app:
    gr.Markdown("# City Analysis & Citizen Services AI")
    ...
app.launch(share=True)


---

6. Example Usage

Input (City Analysis):

Mumbai

Output:

Crime Index: Moderate  
Traffic Safety: High accident risk in peak hours  
Overall Safety: Requires caution in crowded areas, better in residential zones.


---

Input (Citizen Services):

What are the eligibility criteria for free public healthcare?

Output:

Eligibility is determined based on income levels, residency status, and government ID proof. Low-income families and senior citizens receive priority benefits.


---

7. Future Improvements

Add real-time data integration (crime/accident stats from APIs)

Enhance multi-language support

Store query history & allow PDF export of reports

Deploy on cloud platforms (AWS, GCP, Hugging Face Spaces)





