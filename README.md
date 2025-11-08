# LLM Bidding Markets Sequential Auctions for Query Routing and Price Optimization
## About the Project
This project models query routing as a sequential auction, where each user query triggers a bidding process among available LLMs. Each LLM places a bid representing the price it charges and the user selects the LLM that maximizes their quasilinear utility, i.e., the expected value of the response minus the cost. The value is assumed to be stochastic and possibly unknown beforehand.

The platform mediating these interactions seeks to optimize both user utility and overall profit, subject to strategic model behavior, possibly limited budgets, and learning dynamics. We investigate theoretical models and simulation-based implementations of such a marketplace, drawing insights from mechanism design, online learning, and reinforcement learning.

## Theory behind the project

## Implementation of the project
For simplicity we have used only 2 LLMs to illustrate the concept. These LLMs are     
      i.  Model-1 [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)    
      ii. Model-2 [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)    
