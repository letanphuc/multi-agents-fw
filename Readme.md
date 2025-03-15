# Multi-Agent System Framework

This project is a framework for developing a multi-agent system based on [Langchain](https://python.langchain.com/) and [OpenAI LLM](https://platform.openai.com/). Each agent is implemented as a node in a graph, allowing agents to interact and call each other to complete tasks.

## Features

- **Modular Agents**: Each agent functions independently and can collaborate dynamically.
- **Graph-Based Execution**: Agents are connected in a graph structure for complex workflows.
- **Integration with Langchain & OpenAI**: Leverage powerful LLM capabilities for reasoning and task completion.

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Getting Started

Start with the `sample_oop.py` file, which provides a basic implementation of an agent.

### Run the Sample

```bash
python sample_oop.py
```

## How It Works

1. **Define an Agent**: Implement your agent in the `agents/` directory (Example `agents/chart.py`).
2. **Register Agents**: Connect agents into a graph structure.
3. **Run a Task**: One agent can call others to complete subtasks.

## Contribution

Feel free to contribute by submitting pull requests or reporting issues.
