import ollama
import requests
import json
model = "llama3.2:1b"

#### Task 1: Interact with deployed LLM via python 


print("=== START TASK 1.1: Simple HTTP Request ===")
# Simple HTTP Request via requests

# Define the URL of the deployed LLM ( this port is forwarded from the docker container to the host system)
url = "http://localhost:11434/api/generate"

# Define the prompt as json
body_json = {
    "model": model,
    "prompt": "Describe Generative AI in two sentences."
}

# ADD HERE YOUR CODE
# HINT: Send the POST request using the json body
response = requests.post(url, json=body_json) # <-- LÖSUNG

# Check if the request was successful
if response.status_code == 200:
    # Process the response
    response_text = response.text

    # Convert each line to json
    response_lines = response_text.splitlines()
    response_json = [json.loads(line) for line in response_lines]
    for line in response_json:
        # Print the response. No line break
        print(line["response"], end="")
else:
    print("Error:", response.status_code)
print("\n=== END TASK 1.1 ===\n")


# Task Description:
# 2. Use Ollama python library to interact with the LLM: [How To](https://pypi.org/project/ollama/)
# - First use method ``ollama.chat(...)``
# - First use method ``ollama.chat(...)`` with ``stream=True``

print("=== START TASK 1.2: API Call via ollama ===")
# API Call via ollama

# ADD HERE YOUR CODE
# <-- LÖSUNG
response = ollama.chat(
    model=model, 
    messages=[
        {'role': 'user', 'content': 'Describe Generative AI in two sentences.'}
    ]
)


print(response["message"]["content"])
print("=== END TASK 1.2 ===\n")


print("=== START TASK 1.3: Streaming API Call via ollama ===")
# Streaming API Call via ollama

# Response streaming can be enabled by setting stream=True, 
# modifying function calls to return a Python generator where each part is an object in the stream.

# ADD HERE YOUR CODE
# <-- LÖSUNG
stream = ollama.chat(
    model=model,
    messages=[
        {'role': 'user', 'content': 'Describe Generative AI in two sentences.'}
    ],
    stream=True
)

for chunk in stream:
  print(chunk["message"]["content"], end="", flush=True)
print("\n=== END TASK 1.3 ===\n")


#### Task 2: Experimenting with Prompt Techniques

print("=== START TASK 2: Prompt Techniques ===")
# Task Description:
# 1. Create three prompts for a sentiment analysis task: a Zero Shot prompt, a One Shot prompt, and a Few Shot prompt. Use the examples from the table above.
# 2. Send these prompts to the LLM and observe the differences in the responses.
# 3. Compare and discuss the responses.

# ADD HERE YOUR PROMPTS

# <-- LÖSUNG (Zero-Shot)
zero_shot_prompt = """Classify the sentiment of the following text:
Text: 'I absolutely love this new phone!'
Sentiment:"""

# <-- LÖSUNG (One-Shot)
one_shot_prompt = """Classify the sentiment of the text. Here is one example:
Text: 'The food was terrible.'
Sentiment: Negative

Now, classify this text:
Text: 'I absolutely love this new phone!'
Sentiment:"""

# <-- LÖSUNG (Few-Shot)
few_shot_prompt = """Classify the sentiment of the text. Here are a few examples:
Text: 'The food was terrible.'
Sentiment: Negative

Text: 'The movie was decent, not great.'
Sentiment: Neutral

Text: 'What an amazing concert!'
Sentiment: Positive

Now, classify this text:
Text: 'I absolutely love this new phone!'
Sentiment:"""

# Stream the responses and print them
for idx, prompt in enumerate([zero_shot_prompt, one_shot_prompt, few_shot_prompt]):
    prompt_type = ["Zero-Shot", "One-Shot", "Few-Shot"][idx]
    print(f"\n--- {prompt_type} Prompt ---\n")
    print(f"User Prompt:\n{prompt}\n")
    
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    
    print("Model Output:")
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print("\n-----------------------------\n")

print("=== END TASK 2 ===\n")


#### Task 3: Prompt Refinement and Optimization

print("=== START TASK 3: Prompt Refinement ===")
# Objective: Refine a prompt to improve the clarity and quality of the LLM's response.
# Task Description:
# - Start with a basic prompt asking the LLM to summarize a paragraph.
# - Refine the prompt by adding specific instructions to improve the summary's quality. (Example: define how long the summary should be, define on which to focus in the summary)

# Original prompt
original_prompt = "Summarize the following paragraph: Generative AI is a field of artificial intelligence focused on creating new content based on patterns learned from existing data. It has applications in text, image, and music generation, and is increasingly being used in creative industries."

# ADD HERE YOUR PROMPT
# <-- LÖSUNG
refined_prompt = """Summarize the following paragraph in exactly one sentence, focusing on its main applications: 
'Generative AI is a field of artificial intelligence focused on creating new content based on patterns learned from existing data. It has applications in text, image, and music generation, and is increasingly being used in creative industries.'"""

# Stream the responses and print them
for idx, prompt in enumerate([original_prompt, refined_prompt]):
    prompt_type = ["Original Prompt", "Refined Prompt"][idx]
    print(f"\n--- {prompt_type} ---\n")
    print(f"User Prompt:\n{prompt}\n")
    
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    
    print("Model Output:")
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print("\n-----------------------------\n")

print("=== END TASK 3 ===\n")


#### [Optional] Task 4: Structured Prompting with Roles (Pirate Theme)

print("=== START TASK 4: Pirate Prompt ===")
# Objective:
# Learn how to use structured prompts that combine role assignment, clear instructions, and examples to improve the output of language models. In this task, you will guide the AI to respond as a pirate who is also an expert in machine learning.
# Instructions:
# - Role Assignment: In your prompt, specify the role of the AI as a Machine Learning Expert who speaks like a pirate.
# - Instruction: Clearly state what you want the AI to explain or discuss in pirate language.
# - Examples: Provide examples to guide the AI in using pirate lingo while explaining technical concepts.

# Combined Techniques Prompt with Pirate Theme

# <-- LÖSUNG
structured_prompt = """
Ahoy, matey! Ye are a salty sea dog, a Pirate Captain, but ye also be a brilliant Machine Learning Expert.
Yer task is to explain a technical concept to yer crew, using yer pirate lingo.

Here be some examples of yer talk:
- 'Hyperparameters' be 'the settings for yer cannons'.
- 'Overfitting' be like 'knowin' only one treasure map and bein' useless on other seas'.

Now, explain to me... what in Neptune's beard is a 'Neural Network'?
"""

# Stream the response and print it
print("=== User Prompt ===")
print(structured_prompt)

stream = ollama.chat(
    model=model,
    messages=[{"role": "user", "content": structured_prompt}],
    stream=True,
)

print("\n=== Model Output =====")
for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
print("\n")
print("=== END TASK 4 ===\n")