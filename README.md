# Langchain Repository: A Guide to Building Chatbots and Structured Prompt Systems

Welcome to the **Langchain Repository**, a comprehensive project that integrates various large language models (LLMs) to build chatbots, generate itineraries, and provide structured outputs such as JSON and Pydantic models. This repository demonstrates how to use different LLM integrations with LangChain to create dynamic and static prompts, structured outputs, and pipeline workflows.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Usage](#usage)  
   - [Chat Models](#chat-models)  
   - [Prompts](#prompts)  
   - [Structured Outputs](#structured-outputs)  
5. [File Descriptions](#file-descriptions)  
6. [Examples](#examples)  
7. [Contributing](#contributing)  
8. [License](#license)  

---

## Overview

This repository showcases the use of LangChain to work with multiple language models, including OpenAI's GPT, Groq's Llama-based models, and Hugging Face's transformers. It provides examples for using these models to handle tasks like text generation, itinerary planning, and chatbot creation. Additionally, it demonstrates how to process structured outputs using formats like JSON and Pydantic.

---

## Features

- **Integration with Different LLMs:**
  - OpenAI's GPT models
  - Groq's Llama-based models
  - Hugging Face's hosted and local models

- **Prompt Engineering:**
  - Static and dynamic prompts
  - Template-based input generation

- **Structured Output Processing:**
  - JSON output parsing
  - TypedDict and Pydantic-based outputs

- **User Interfaces:**
  - Streamlit-based front-end for user interaction

- **Examples of Pipeline Workflows:**
  - Combining models and parsers for complex outputs
  - Multi-step workflows with chained templates and parsers

---

## Installation

### Prerequisites

- Python 3.8 or higher
- A virtual environment manager (e.g., `venv`, `conda`)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Langchain.git
   cd Langchain
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add your model API keys and necessary credentials:
     ```
     OPENAI_API_KEY=your_openai_key
     GROQ_API_KEY=your_groq_key
     HUGGINGFACE_API_KEY=your_huggingface_key
     ```

---

## Usage

### 1. Chat Models

The repository provides multiple implementations of chat models from different providers. Each implementation demonstrates how to invoke a model and retrieve responses. Below are the available examples:

#### OpenAI's ChatGPT
File: `./repos/Langchain/ChatModels/OpenAi_chatmodel.py`  
```py
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model='gpt-4o-2024-08-06')
result = model.invoke("What is the capital of Delhi?")
print(result)
```

#### Groq's Llama-Based Models
File: `./repos/Langchain/ChatModels/Groq_cloud_chat_models.py`  
```py
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.3)
result = model.invoke("What is the capital of Delhi?")
print(result.content)
```

#### Hugging Face Hosted Models
File: `./repos/Langchain/ChatModels/3_HuggingFace_api_inference.py`  
```py
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of India?")
print(result.content)
```

### 2. Prompts

This repository supports both static and dynamic prompt templates.

#### Dynamic Prompts
File: `./repos/Langchain/Prompts/Dynamic_prompts.py`  
```py
from langchain_core.prompts import PromptTemplate, load_prompt
# Dynamic prompt example with Streamlit UI for itinerary generation
```

#### Static Prompts
File: `./repos/Langchain/Prompts/Static_prompt.py`  
```py
# A static user-input prompt example using Streamlit
```

### 3. Structured Outputs

The repository demonstrates how to generate and parse structured outputs from models.

#### JSON Output Parsing
File: `./repos/Langchain/Structured_output/json_output_parser.py`  
```py
from langchain_core.output_parsers import JsonOutputParser
# Example using JSON output parsing with a structured itinerary prompt
```

#### Pydantic-Based Output
File: `./repos/Langchain/Structured_output/Pydantic_structure_output.py`  
```py
from pydantic import BaseModel, Field
# Example using Pydantic models to define and parse structured output
```

#### TypedDict-Based Output
File: `./repos/Langchain/Structured_output/type_dict_structure_output.py`  
```py
from typing import TypedDict, Annotated
# Example using TypedDict for parsing structured outputs
```

---

## File Descriptions

### Chat Models

- **`main.py`**: Base script with a simple "Hello, World!" function.
- **`OpenAi_chatmodel.py`**: Example of using OpenAI's ChatGPT.
- **`Groq_cloud_chat_models.py`**: Example of using Groq's Llama-based models.
- **`3_HuggingFace_api_inference.py`**: Example of using Hugging Face's hosted models.
- **`4_HuggingFace_model_locally.py`**: Example of using Hugging Face models locally.

### Prompts

- **`Dynamic_prompts.py`**: Creates dynamic prompts for generating itineraries.
- **`Static_prompt.py`**: Example of a static prompt with user input.
- **`chatbot.py`**: Interactive chatbot using Groq's Llama model.
- **`prompt_template_generator.py`**: Creates and saves prompt templates for reuse.

### Structured Outputs

- **`Pydantic_structure_output.py`**: Uses Pydantic models for structured output.
- **`json_output_parser.py`**: Demonstrates JSON parsing with LangChain.
- **`type_dict_structure_output.py`**: Uses TypedDict to define structured outputs.
- **`str_output_Parser.py`**: Combines multiple templates and parsers in a pipeline.

### Requirements

- **`requirements.txt`**: Contains all necessary Python dependencies.

---

## Examples

### Example 1: Generating an Itinerary with a Dynamic Prompt
Run the following command to launch the Streamlit UI for itinerary generation:

```bash
streamlit run ./repos/Langchain/Prompts/Dynamic_prompts.py
```

### Example 2: Structured Output with Pydantic
Run the following script to generate a structured itinerary using Pydantic:

```bash
python ./repos/Langchain/Structured_output/Pydantic_structure_output.py
```

---

## Contributing

We welcome contributions to this repository! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a clear description of your changes.

---

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Thank you for exploring the Langchain repository! 🚀
