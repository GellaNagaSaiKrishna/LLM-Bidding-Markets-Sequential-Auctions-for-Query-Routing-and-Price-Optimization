# LLM Bidding Markets Sequential Auctions for Query Routing and Price Optimization
## About the Project
This project models query routing as a sequential auction, where each user query triggers a bidding process among available LLMs. Each LLM places a bid representing the price it charges and the user selects the LLM that maximizes their quasilinear utility, i.e., the expected value of the response minus the cost. The value is assumed to be stochastic and possibly unknown beforehand.

The platform mediating these interactions seeks to optimize both user utility and overall profit, subject to strategic model behavior, possibly limited budgets, and learning dynamics. We investigate theoretical models and simulation-based implementations of such a marketplace, drawing insights from mechanism design, online learning, and reinforcement learning.

## Theory behind the project
We conduct an auction between all the LLMs, and we chose an LLM such that our reward is maximized(greedy approach). In each step we compute the expected reward using the UCB Algorithm, and chose the best LLM with Reverse Myerson Auctions. For Mathematical derivation of Myerson Auction refer [here](https://drive.google.com/file/d/1Hu6liiz3RabZPDYA2xIP6wkNT6pMUHMO/view?usp=sharing).
We pay the LLM the min(upper bound(reservation price),next_lowest_bid). If all LLM have cost more than the reservation price we skip the round.

## Implementation of the project
For simplicity we have used only 2 LLMs to illustrate the concept. These LLMs are     
      i.  Model-1 [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)    
      ii. Model-2 [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)    
To test the model we have chosen a simple City Capital dataset from the [kaggle](https://www.kaggle.com/datasets/dataanalyst001/all-capital-cities-in-the-world)  
In each round we give input as country name and our mechanism choses the best llm and makes it output Capital, Continent, Latitude and Longitude. After each round we check the answer outputed from LLM with answers in the dataset and update the accuracies of the LLM.

