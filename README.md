# Personalized Medication Assistant

## Overview
This is a Gemini-based smart assistant that uses RAG to answer user queries about their medication regime based on a personalized profile stored using ChromaDB.<br><br>
An example of typical usage is shown below:

![Usage Example](/images/Screenshot_01.png)

## Install 
Execute the follwing command to install dependancies:
```pip install -r requirements.txt```

## Run
1. Run in terminal: `python generate_database.py`. This creates a sample database.
2. Run in terminal: `python main.py`. This launches a Gradio UI when queries can be input.
3. Navigate to [http://127.0.0.1:7860/](http://127.0.0.1:7860/) to access the Gradio UI.
