# LLM Bidding Markets Sequential Auctions for Query Routing and Price Optimization
This project models query routing as a sequential auction, where each user query triggers a bidding process among available LLMs. Each LLM places a bid representing the price it charges and the user selects the LLM that maximizes their quasilinear utility, i.e., the expected value of the response minus the cost. The value is assumed to be stochastic and possibly unknown beforehand.

The platform mediating these interactions seeks to optimize both user utility and overall profit, subject to strategic model behavior, possibly limited budgets, and learning dynamics. We investigate theoretical models and simulation-based implementations of such a marketplace, drawing insights from mechanism design, online learning, and reinforcement learning.

