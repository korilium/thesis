#!/usr/bin/env python
# coding: utf-8

# # Reinforcement Learning 

# Supervised and unsupervised learning are the two most widely studied and researched branches of Machine Learning (ML). Besides these two, there is also a third subcategorie in ML called Reinforcement Learning (RL). The three branches have fundamental differences between eachother. Supervised learning for example is designed to learn from a training set of labeled data, where each element of the training set describes a certain situation and is linked to a label/action the supervisor has provided {cite}`hammoudeh2018concise`. RL on the other hand is a method in which the machine tries to map situation to actions by maximizing a reward signal {cite}`arulkumaran2017brief`. The two methods are fundementally different from each other on the fact that in RL there is no supervisor which provides the label/action the machine needs to take, rather there is a reward system set up from which the machine can learn the correct action/label {cite}`hammoudeh2018concise`. contrarily to supervised learning, unsupervised learning tries to find hidden structures within an unlabeled dataset. This might seem similar to RL as both methods work with unlabeled datasets, but RL tries to maximize a reward signal instead of finding only hidden structures in the data {cite}`arulkumaran2017brief`. 
# 
# RL finds it roots in multiple research fields. Each of these fields contributes to the RL in its own unique way (see figure) {cite}`hammoudeh2018concise`. For example,  RL is similar to natural learning processes where the method of learning is by experiencing many failures and successes. Therefore psychologists have used RL to mimic psychological processes when an organism makes choices based on experienced rewards/punishments {cite}`eckstein2021reinforcement`. While psychologists are mimicing psychological processes, Nueroscientists are using RL to focus on a well-defined network or regions of the brain that implement value learning {cite}`eckstein2021reinforcement`. 

# ```{figure} C:/Users/ignac/Documents/GitHub/thesis/notebook/images/tree.png
# ---
# height: 250px
# width: 600px
# name: tree-fig
# ---
# research fields involved in reinforcement learning
# ```

# ## Finite Markov Decision Processes

# RL can be represented in finite Markov decision processes (MDPs), which are classical formalizations of sequantial decision making. More specifically, MPDs give rise to a structure in which delayed rewards can be balanced with immediate rewards {cite}`sutton2018reinforcement`. It also enables a straightforward framing of learning from interaction to achieve a goal {cite}`levine2018reinforcement`. In it's most simplest form RL works with an Agent-Environment Interface. The agent is exposed to some representation of the environment's state $S_t \in \mathrm{S}$. From this representation the agent needs to chose an action $ A_t \in \mathcal{A}(s)$, which will result in a numerical reward $R_{t+1} \in \mathrm{R} $ and a new state $S_{t+1}$ (see figure 2) {cite}`sutton2018reinforcement`. The goal for the agent is to learn a mapping from states to action called a policy $\pi$ that maximizes the expected rewards:
# 
# $$ \pi^* = argmax_{\pi} E[R|\pi] $$
# 
# If the MPDs is finite and discrite, the sets of states, actions and rewards ($S$, $A$ , and $R$) all have a finite number of elements. The agent-environment interaction can then be subdivided into episode {cite}`arulkumaran2017brief`.  The agent's goal is to maximize the expected discounted cumulative return in the episode {cite}`franccois2018introduction`: 
# 
# $$ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... + \gamma^{T-t-1}R_T = \sum_{k=0}^T \gamma^k R_{t+k+1}$$ 
# 
# Where T indicates the terminal state and $\gamma$ is the discount rate. The terminal state $S_T$ is mostly followed by a reset to a starting state or sample from a starting distribution of states {cite}`franccois2018introduction`. An episode ends once the reset has occured. The discount rate represents the present value of future rewards. If $\gamma = 0$, the agent is myopic and is only concerned in maximizing the immediate rewards. The agent can consequently be considerd greedy {cite}`sutton2018reinforcement`. 
# 
# The returns can be rewritten in a dynammic programming approach:
# 
# $$ G_t = R_{t+1} + \gamma(R_{t+2} + \gamma R_{t+3} + ... + \gamma^{T-t-2}R_T) $$
# $$ G_t = R_{t+1} + \gamma G_{t+1}$$ 
# 

# ```{figure} C:/Users/ignac/Documents/GitHub/thesis/notebook/images/standard_model.png
# ---
# height: 300px
# width: 600px
# name: standard_model-fig
# ---
# standard model reinforcement learning 
# ```

# A key concept of MPDs is the Markov property: Only the current state affects the next state {cite}`franccois2018introduction`. The random varianbles (RV) $R_t$ and $S_t$ have then well defined discrete probability distributions dependend only on the preceding state and action: 
# 
# $$ p(s', t| s, a) = Pr(S_t = s', R_t = r | S_{t-1} = s, A_{t-1}=a) $$
# 
# For all $s', s \in \mathrm{S} , r \in \mathrm{R}, a \in \mathrm{A}(s) $. The probability of each element in the sets $S$ and $R$ completely chararcterizes the environment {cite}`sutton2018reinforcement`. This can be relaxed by some alogrithms as this is an unrealistic assumption to make. The Partial Observable Markov Decision Process (POMDP) algorithm for example maintains a belief over the current state given the previous belief state, the action taken and the current observation {cite}`arulkumaran2017brief`.  Once $p$ is known, the environment is fully discribed and functions like a transition function $T : D \times A \to p(S)$ and a reward function $R: S \times A \times S \to \R$ can be deducted {cite}`sutton2018reinforcement`.
# 
# Most algorithms in RL use a value function to estimate the value of a given state for the agent. Value functions are defined by the policy $\pi$ the agent has decided to take. As mentioned previously, $\pi$ is the mapping of states to probabilities of selecting an action. The value function $v_{\pi}(s)$ in a state $s$ following a policy $\pi$ is as followes: 
# 
# $$ v_{\pi}(s) = E_{\pi}[G_t | S_t = s] = E_{\pi}[\sum_{k=0}^T \gamma^kR_{t+k+1} | S_t=s] $$
# 
# This can aso be rewritten in a dynammic programming approach: 
# 
# 
# 

# ```{math}
# :label: my_label
# v_{\pi}(s) = E_{\pi}[G_t | S_t = s] \\
# = E_{\pi}[R_{t+1} + \gamma G_{t+1} | S_t = s] \\
# = \sum_a \pi(a|s) \sum_{s'} \sum_r p(s', r|s,a)[r + \gamma E_{pi}[G_{t+1} | S_{t+1} = s'] \\
# = \sum_a \pi(a|s) \sum_{s', r}p(s', r|s,a)[r + \gamma v_{\pi}(s')| S_{t+1} = s'] 
#  ```

#  The formula is called the Bellman equation of $v_{\pi}$. It describes the relationschip between the value of a state and the values of its successor states given a certain policy $\pi$. The relation can also be represented by a backup diagram (see figure 3). If $v_{\pi}(s)$ is the value of a given state, then $q_{\pi}(s,a)$ is the value of a given action of that state: 
# 
#  $$ q_{\pi}(s,a) = E_{\pi}[G_t | S_t = s, A_t = a] = E_{\pi}[\sum_{k=0}^T \gamma^kR_{t+k+1} | S_t=s, A_t = a] $$ 
# 
#  This can be seen in the backup diagram as starting from the black dot and cumputing the subsequential value thereafter. $q_{\pi}(s,a)$ is also called the action-value function as it describes each value of an action for each state. 

# ```{figure} C:/Users/ignac/Documents/GitHub/thesis/notebook/images/backup_diagram.png
# ---
# height: 250px
# width: 300px
# name: backup_diagram-fig
# ---
# General backup diagram 
# ```

# For the agent it is important to find the optimal policy in which it maximizes the expected cumulative rewards. The optimal policy $\pi_*$ is the policy for which $v_{\pi_*}(s) > v_{\pi}(s)$ for all $s \in S$. An optimal policy also has the same action-value function $q_*(s,a)$ for all $s \in S$ and $a \in A$. The optimal policy does not depend soley on one policy and can encompass multiple policy. It is thus not policy dependend: 
# 
# $$ v_*(s) = max_{a \in A(s)} q_{\pi_*}(s,a) $$
# $$ = max_{a} E_{\pi_*}[G_t | S_t=s, A_t=a] $$ 
# $$ = max_{a} E_{\pi_*}[R_{t+1} + \gamma G_{t+1} | S_t=s, A_t=a] $$
# $$ = max_{a} E[R_{t+1} + \gamma v_*(S_{t+1}) | S_t=s, A_t=a] $$
# 
# Once $v_*(s)$ is found, you just need to apply a greedy algorithm as the optimal value function already takes into account the long-term consequences of choosing that action. Finding $q_*(s,a)$ makes things even easier, as the action-value function caches the result of all one-step-ahead searches. 
# 
# Solving the Bellman equation of the value function or the action-value function such that we know each all possibilities with their probabilities and rewards is in most practical cases not possible. Typical due to three main factors. The first problem is obtaining full knowledge of the dynamics of the environment. The second factor is the computational resources to complete the calculation. the last factor is that the states need to have the markov property.   To circumvent these obstacles RL tries to approximate the Bellman optimality equation using various methods. In the next chapter, a brief layout of theser method is discussed with a focus on the methods applicable for financial planning. 

# ##  model-based RL, model-free RL and planning

# A general theory in finding the optimal policy $\pi_*$ is called Generelized Policy Iteration (GLI). This method is applied to almost all RL algorithms. The main idea behind GLI is that there is a process which evaluates the value function of the current policy $\pi$ called policy evaluation and a process which improves the current value function called policy improvement. To find the optimal policy these two processes work in tandem with eachother as seen in figure ... Counterintuitively, these processes also work in a conflicting manner as policy improvement makes the policy incorrect and it is thus no longer the same policy. While policy evaluations creates a consistent policy and thus the policy no longer improves upon itself. This idea runs in parallel with the balance between exploration and exploitation in RL.  If the focus lies more on exploration, the agent frequently tries to find states which improve the value function. However, putting more emphasis on exploration is a costly setting as the agent will more frequenlty choose suboptimal policies to explore the state space. If exploitation is prioritised, the agent will take a long time to find the optimal policy as the agent is likely not to explore new states to improve the policy.  is a good example of the influential balance between exploration and exploitation. 

# ```{figure} C:/Users/ignac/Documents/GitHub/thesis/notebook/images/GPI.png
# ---
# height: 300px
# width: 500px
# name: GPI-fig
# ---
# Generalized policy iteration
# ```

# Reinforcement Learning can be subdivided between model-based RL and model-free RL. In model-free RL the dynamics of the environment are not known. $\pi_*$ is found by purily interacting with the environment. Meaning that these algorithms do not use transition probability distribution and reward function related to MDP. Moreover, model-free RL have irreversible access to the environment. Meaning the algorithm has to move forward after an action is taken. Good examples of model-free RL techniques are the Q-learning and Policy Optimization algorithms. They tend to be used on a variety of tasks, like playing video games to learning complicated locomotion skills. Model-free RL lay at the fundation of RL and are one of the first algorithms to be applied in RL. On the other hand, model-based RL is developed independently and in parallel with planning methods like optimal control and the search community as they both solve the same problem but differ in the approach. Most algorithms in model-based RL have a model which describes the dynamics of the environment. This enables the agent to think in advance and as it were plan for possible actions. Model-based reinforcement learning finds thus large similarities with the Planning literature and as a result a lot of cross breeding between the two is happening. For example an extension of the POMP algorithm called Partially Observable Multi-Heuristic Dynammic Programming (POMHDP) is based on recent progress from the search community. A hybrid version of the two approaches in which the model is learned through interaction with the environment has also been widely applied. The imagination-augmented agents (12A) for example combines model-based and model-free aspects by employing the predictions as additional context in a deep policy network.  In the next subsection three fundamental algorithms in RL are discussed which will enable us to better capture the dimensions and challenges of a RL algorithm.
# 

# ### Dynammic Programming, Monte Carlo Methods and Temporal-Difference Learning

# Dynammic Programming (DP) is known for two algorithms in RL: value iteration (VI) and policy iteration (PI). For both methodes the dynamics of the environment need to be completly known and they therefore fall under model-based RL. The two algorithms also use a discrete time, state and action MDP as they are iterative procedures. The PI can be subdivided into three steps: initialize, policy evaluation and policy improvement. The first step is to initialize the value function $v_{\pi}$ by choosing an arbitrary policy $\pi$. The following step is to evaluate the function successively by updating the the Bellman equation eq 2.1 . Updating on the Bellman equation is also called the expected update as the equation is updated using the whole state space instead of a sample of the state space. One update is also called a sweep as the update sweeps through the state space. Now that we have updated the value function $v_{\pi}$, we know how good it is to follow the current policy. The next step is to deviate from the policy trajectory and chose a different action a in state s to find a more optimal policy value. We compute the new $\pi '$ and compare it to the old policy. The new policy is accepted if $\pi '(s) > \pi(s)$. This process is repeated untill a convergence criteria is met. The complete algorithm can be found in the appendix. VI combines the policy evaluation with the policy improvement by truncating the sweep with one update of each state. It effectivily combines the policy evaluation and policy evaluation in one sweep (see appendix for algorithm). PI and VI are the foundation of DP and numerous adaptions have been made on these algorithms. For example have... . Adaptive Dynammic programming is  
# 
#     
# 

# The Monte Carlo (MC) methods do not assume full knowledge of the dynamics of the environment and are thus considered model-free RL techniques. They only require a sample sequence of states, actions and rewards from interaction of an environment. Techniquely, a model is still required which generates sample transitions, but the complete probability distribtion $p$ of the dynammic system is not neccesary. The idea behind almost all MC methods is that the agent learns the optimal policy by averaging the sample returns of a policy $\pi$. They can therefore not learn on an online basis as after each episode they need to average their returns. Another difference between the two methods is that the MC method does not bootstrap like DP. Meaning, each state has an independed estimate. Note that Monte Carlo methods create a nonstationary problem as each action taken at a state depends on the previous states. MC methods can either estimate  a state value (eq) or  estimate the value of a state-action pairs (eq) (recall that the state-action values are the value of an action given a state). If state values are estimated, a model is required as it needs to be able to look ahead one step and choose the action which leads to the best reward and next state. With action value estimation you already estimated the value of the action and no model needs to be taken into account.  Monte Carlo methods also use a term called visits. A visit is when a state or state-action pair is in the sample path. Multiple visits to a state are possible in an episode. Two general Monte Carlo Methods can be deducted from visits. The every-visit MC methods and the first-visit MC methods. The every-visit MC methods estimates the value of a state as the average of the returns that have followed all visits to it. The first visit method only looks at the first visit of that state to estimate the average returns. The biggest hurdle in MC methods is that most state-action pairs might never be visited in the sample.
# 
# To overcome this problem multiple solutions have been explored. The naïve solution to this problem is called the exploring starts. Here, the idea is to allocate to each action in each state a nonzero probability at the start of the process. Although this is not possible in a practical setting where we truly want to interact with an environment, it enables us to improve to policy by making it greedy with respect to the current value function. If an infinite number of episodes are taken, the policy improvement theory states that the policy $\pi$ will convergence too the optimal policy $\pi_*$ given the exploring starts. The other two possibilities are on-policy methods and off-policy methods. On-policy methods attempt to improve on the current policy. This is also called a soft policy as $\pi(a|s) > 0$ for all $s \in S$ and all $ a \in A(s)$, but shifts eventual to the deterministic optimal policy. One of these on-policy methods uses a $\varepsilon$-greedy policy. The $\varepsilon$-greedy policy uses with probability $\varepsilon$ a random action instead of the greedy action. A pseudocode of on-policy first visit MC for $\varepsilon$-soft policies algorithm can be found in the appendix. 
# 
# 
# 

# 
# ```{figure} C:/Users/ignac/Documents/GitHub/thesis/notebook/images/state_space.png
# ---
# height: 350px
# width: 500px
# name: state_space-fig
# ---
# state_space dimensions  
# ```
# 

# 
# ```{figure} C:/Users/ignac/Documents/GitHub/thesis/notebook/images/breadth_and_dept.png
# ---
# height: 400px
# width: 600px
# name: breadth_and_dept-fig
# ---
# breadth and dept dimensions  
# ```

# ## Challenges in RL and deep RL

# ## Reinforcement learning and financial planning 
