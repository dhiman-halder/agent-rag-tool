# AI Agent RAG Tool

This project demonstrates the use of LangGraph to create a agent that integrates with tools like Tavily search and a Vector DB retriever.

## Requirements
- Python 3.8 or higher
- OpenAI API Key
- Tavily API Key

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/dhiman-halder/agent-rag-tool.git
   cd agent

2. **Set Up a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Mac/Linux
    # For Windows:
    # venv\Scripts\activate

3. **Install Dependencies**
    Install the required Python packages using `pip`:
    ```bash
    pip install -r requirements.txt

4. **Set Up Environment Variables**
    Create a `.env` file in the project root and add the following:
    ```bash
    OPENAI_API_KEY=<your_openai_api_key>
    TAVILY_API_KEY=<your_tavily_api_key>

## Running the Application
1. Run the Script Execute the `app.py` file to start the agent:
   ```bash
   python app.py 
2. Agent Interaction The agent will process the predefined input message and stream responses in real-time.

## License
This project is licensed under the MIT License. See the LICENSE file for details.