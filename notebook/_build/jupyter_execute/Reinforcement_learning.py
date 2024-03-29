#!/usr/bin/env python
# coding: utf-8

# # Reinforcement Learning 

# Supervised and unsupervised learning are the two most widely studied and researched branches of Machine Learning (ML). Besides these two, there is also a third subcategory in ML called Reinforcement Learning (RL). The three branches have fundamental differences between each other. Supervised learning for example is designed to learn from a training set of labeled data, where each element of the training set describes a certain situation and is linked to a label/action the supervisor has provided {cite}`hammoudeh2018concise`. On the other hand, RL is a method in which the machine tries to map situations to actions by maximizing a reward signal {cite}`arulkumaran2017brief`. The two methods are fundamentally different from each other in the fact that in RL there is no supervisor which provides the label/action the machine needs to take, rather there is a reward system set up from which the machine can learn the correct action/label {cite}`hammoudeh2018concise`. contrarily to supervised learning, unsupervised learning tries to find hidden structures within an unlabeled dataset. This might seem similar to RL as both methods work with unlabeled datasets, but RL tries to maximize a reward signal instead of finding only hidden structures in the data {cite}`arulkumaran2017brief`. Unsupervised learning on the other hand learns about how the data is distributed {cite}`shin2019reinforcement`.
# 
# 
# 
# RL finds its roots in multiple research fields. Each of these fields contributes to the RL in its own unique way (see {numref}`Figure {number} <tree-fig>`) {cite}`hammoudeh2018concise`. For example,  RL is similar to natural learning processes where the learning method is by experiencing many failures and successes. Therefore psychologists have used RL to mimic psychological processes when an organism makes choices based on experienced rewards/punishments {cite}`eckstein2021reinforcement`. While psychologists are mimicking psychological processes, neuroscientists are using RL to focus on a well-defined network of regions of the brain that implement value learning {cite}`eckstein2021reinforcement`. 

# ```{figure} C:/Users/ignac/Documents/GitHub/thesis/notebook/images/tree.png
# ---
# height: 250px
# width: 600px
# name: tree-fig
# ---
# research fields involved in reinforcement learning
# ```

# ## Finite Markov Decision Processes

# RL can be represented in finite Markov decision processes (MDPs), which are classical formalizations of sequential decision making. More specifically, MPDs give rise to a structure in which delayed rewards can be balanced with immediate rewards {cite}`sutton2018reinforcement`. It also enables a straightforward framing of learning from interaction to achieve a goal {cite}`levine2018reinforcement`. In its simplest form, RL works with an Agent-Environment Interface. The agent is exposed to some representation of the environment's state $S_t \in \mathrm{S}$. From this representation the agent needs to chose an action $ A_t \in \mathcal{A}(s)$, which will result in a numerical reward $R_{t+1} \in 	\mathbb{R} $ and a new state $S_{t+1}$ (see {numref}`Figure {number} <standard_model-fig>`)  {cite}`sutton2018reinforcement`. The goal for the agent is to learn a mapping from states to action called a policy $\pi$ that maximizes the expected rewards:
# 
# $$ \pi^* = argmax_{\pi} E[R|\pi] $$
# 
# If the MPDs are finite and discrete, the sets of states, actions ,and rewards ($S$, $A$ , and $R$) all have a finite number of elements. The agent-environment interaction can then be subdivided into episode {cite}`arulkumaran2017brief`.  The agent's goal is to maximize the expected discounted cumulative return in the episode {cite}`franccois2018introduction`: 
# 
# ```{math}
# :label: return
# G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... + \gamma^{T-t-1}R_T = \sum_{k=0}^T \gamma^k R_{t+k+1}
# ```
# 
# Where T indicates the terminal state and $\gamma$ is the discount rate. The terminal state $S_T$ is often followed by a reset to a starting state or sample from a starting distribution of states {cite}`franccois2018introduction`. An episode ends once the reset has occurred. The discount rate represents the present value of future rewards. If $\gamma = 0$, the agent is myopic and is only concerned with maximizing the immediate rewards. The agent can consequently be considered greedy {cite}`shin2019reinforcement`. 
# 
# The returns can be rewritten in a dynamic programming approach:
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

# A key concept of MPDs is the Markov property: Only the current state affects the next state {cite}`franccois2018introduction`. The random variables (RV) $R_t$ and $S_t$ have then well defined discrete transition probability distributions dependent only on the previous state and action: 
# 
# $$ p(s', t| s, a) = Pr(S_t = s', R_t = r | S_{t-1} = s, A_{t-1}=a) $$
# 
# For all $s', s \in \mathrm{S} , r \in 	\mathbb{R}, a \in \mathrm{A}(s) $. The probability of each element in the sets $S$ and $R$ completely characterizes the environment {cite}`sutton2018reinforcement`. This is an unrealistic assumption to make, and several algorithms relax the Markov property. The Partial Observable Markov Decision Process (POMDP) algorithm, for example, maintains a belief over the current state given the previous belief state, the action taken, and the current observation {cite}`arulkumaran2017brief`.  Once $p$ is known, the environment is fully described and functions like a transition function $T : D \times A \to p(S)$ and a reward function $R: S \times A \times S \to \mathbb{R}$ can be deducted {cite}`sutton2018reinforcement`.
# 
# Most algorithms in RL use a value function to estimate the value of a given state for the agent. Value functions are defined by the policy $\pi$ the agent has decided to take. As mentioned previously, $\pi$ is the mapping of states to probabilities of selecting an action. The value function $v_{\pi}(s)$ in a state $s$ following a policy $\pi$ is as follows: 
# ```{math}
# :label: value
# v_{\pi}(s) = E_{\pi}[G_t | S_t = s] = E_{\pi}[\sum_{k=0}^T \gamma^kR_{t+k+1} | S_t=s]
# ```
# This can also be rewritten in a dynamic programming approach: 
# 
# 
# 

# $$v_{\pi}(s) = E_{\pi}[G_t | S_t = s] $$
# $$ \hspace{2.4cm}= E_{\pi}[R_{t+1} + \gamma G_{t+1} | S_t = s] $$
# $$  \hspace{6cm} = \sum_a \pi(a|s) \sum_{s'} \sum_r p(s', r|s,a)[r + \gamma E_{pi}[G_{t+1} | S_{t+1} = s'] $$
# 
# ```{math}
# :label: BELL
# \hspace{5.8cm}= \sum_a \pi(a|s) \sum_{s', r}p(s', r|s,a)[r + \gamma v_{\pi}(s')| S_{t+1} = s'] 
# ```

#  The formula is called the Bellman equation of $v_{\pi}$. It describes the relationship between the value of a state and the values of its successor states given a certain policy $\pi$. The relation can also be represented by a backup diagram (see {numref}`Figure {number} <backup_diagram-fig>`). If $v_{\pi}(s)$ is the value of a given state, then $q_{\pi}(s,a)$ is the value of a given action of that state: 
# 
# ```{math}
# :label: state-action
#  q_{\pi}(s,a) = E_{\pi}[G_t | S_t = s, A_t = a] = E_{\pi}[\sum_{k=0}^T \gamma^kR_{t+k+1} | S_t=s, A_t = a] 
# ```
# 
#  This can be seen in the backup diagram as starting from the black dot and computing the subsequential value thereafter. $q_{\pi}(s,a)$ is also called the action-value function as it describes each value of an action for each state. 

# ```{figure} C:/Users/ignac/Documents/GitHub/thesis/notebook/images/backup_diagram.png
# ---
# height: 250px
# width: 300px
# name: backup_diagram-fig
# ---
# General backup diagram 
# ```

# For the agent, it is important to find the optimal policy which maximizes the expected cumulative rewards. The optimal policy $\pi_*$ is the policy for which $v_{\pi_*}(s) > v_{\pi}(s)$ for all $s \in S$. An optimal policy also has the same action-value function $q_*(s,a)$ for all $s \in S$ and $a \in A$. The optimal policy does not depend solely on one policy and can encompass multiple policies. It is thus not policy dependent: 
# 
# $$ v_*(s) = max_{a \in A(s)} q_{\pi_*}(s,a) $$
# $$ = max_{a} E_{\pi_*}[G_t | S_t=s, A_t=a] $$ 
# $$ = max_{a} E_{\pi_*}[R_{t+1} + \gamma G_{t+1} | S_t=s, A_t=a] $$
# $$ = max_{a} E[R_{t+1} + \gamma v_*(S_{t+1}) | S_t=s, A_t=a] $$
# 
# Once $v_*(s)$ is found, you  need to apply a greedy algorithm as the optimal value function already takes into account the long-term consequences of choosing that action. Finding $q_*(s,a)$, makes things even easier, as the action-value function caches the result of all one-step-ahead searches. 
# 
# Solving the Bellman equation of the value function or the action-value function such that we know all possibilities with their probabilities and rewards is in most practical cases impossible. Typical due to three main factors {cite}`sutton2018reinforcement`. The first problem is obtaining full knowledge of the dynamics of the environment. The second factor is the computational resources to complete the calculation. The last factor is that the states need to have the Markov property. To circumvent these obstacles, RL tries to approximate the Bellman optimality equation using various methods. In the next chapter, a brief layout of these methods is discussed, focussing on the methods applicable for financial planning. 

# ##  Generelized Policy Iteration, Model-based RL and Model-free RL 

# A general theory in finding the optimal policy $\pi_*$ is called Generalized Policy Iteration (GLI). This method is applied to almost all RL algorithms. The main idea behind GLI is that there is a process that evaluates the value function of the current policy $\pi$ called policy evaluation and a process that improves the current value function called policy improvement {cite}`sutton2018reinforcement`, {cite}`van2012reinforcement`. To find the optimal policy these two processes work in tandem with each other as seen in {numref}`Figure {number} <GPI-fig>` {cite}`van2012reinforcement`. Counterintuitively, these processes also work in a conflicting manner as policy improvement makes the policy incorrect and it is thus no longer the same policy {cite}`sutton2018reinforcement`,  {cite}`barreto2020fast`. While policy evaluations create a consistent policy and thus the policy no longer improves upon itself. This idea runs in parallel with the balance between exploration and exploitation in RL.  If the focus lies more on exploration, the agent frequently tries to find states which improve the value function. However, putting more emphasis on exploration is a costly setting as the agent will more frequently choose suboptimal policies to explore the state space. If exploitation is prioritized, the agent will take a long time to find the optimal policy as the agent is likely not to explore new states to improve the policy {cite}`van2012reinforcement`.  An $\varepsilon$-greedy algorithm is a good example of an algorithm where the balance between exploration and exploitation is important. 

# ```{figure} C:/Users/ignac/Documents/GitHub/thesis/notebook/images/GPI.png
# ---
# height: 300px
# width: 500px
# name: GPI-fig
# ---
# Generalized policy iteration
# ```

# Reinforcement Learning can be subdivided between model-based RL and model-free RL. In model-free RL the dynamics of the environment are not known. $\pi_*$ is found by purely interacting with the environment. Meaning that these algorithms do not use transition probability distribution and reward function related to MDP {cite}`franccois2018introduction`. Moreover, model-free RL has irreversible access to the environment. Meaning the algorithm has to move forward after an action is taken {cite}`moerland2020model`. Model-based RL on the other hand has reversible access to the environment because they can revert the model and make another trail from the same state {cite}`moerland2020framework`. Good examples of model-free RL techniques are the Q-learning and Sarsa. They tend to be used on a variety of tasks, like playing video games to learning complicated locomotion skills {cite}`sutton2018reinforcement`. Model-free RL lay at the foundation of RL and are the first algorithms to be applied in RL were model-free RL techniques. On the other hand, model-based RL is developed independently and in parallel with planning methods like optimal control and the search community as they both solve the same problem but differ in the approach {cite}`wang2019exploration`. Most algorithms in model-based RL have a model which describes the dynamics of the environment. They sample from that model to then improve a learned value or policy function {cite}`moerland2020framework` (see {numref}`Figure {number} <model_based_RL>`). This enables the agent to think in advance and as it were plan for possible actions. Model-based reinforcement learning finds thus large similarities with the Planning literature and as a result, a lot of cross-breeding between the two is happening. For example, an extension of the POMP algorithm called Partially Observable Multi-Heuristic Dynamic Programming (POMHDP) is based on recent progress from the search community {cite}`kim2019pomhdp`. A hybrid version of the two approaches in which the model is learned through interaction with the environment, has also been widely applied. The imagination-augmented agents (12A) for example combines model-based and model-free aspects by employing the predictions as an additional context in a deep policy network {cite}``moerland2020model.  In the next subsection, three fundamental algorithms in RL are discussed which will enable us to better capture the dimensions and challenges of RL algorithms.
# 

# ```{figure} C:/Users/ignac/Documents/GitHub/thesis/notebook/images/model_based_RL.png
# ---
# height: 200px
# width: 300px
# name: model_based_RL
# ---
# Model-based Reinforcement Learning 
# ```

# ### Dynamic Programming, Monte Carlo Methods and Temporal-Difference Learning

# Dynamic Programming (DP) is known for two algorithms in RL: value iteration (VI) and policy iteration (PI). For both methods, the dynamics of the environment need to be completely known and they ,therefore, fall under model-based RL. The two algorithms also use a discrete time, state and action MDP as they are iterative procedures. The PI can be subdivided into three steps: initialize, policy evaluation and policy improvement {cite}`van2012reinforcement`. The first step is to initialize the value function $v_{\pi}$ by choosing an arbitrary policy $\pi$. The following step is to evaluate the function successively by updating the Bellman equation {eq}`BELL`. Updating on the Bellman equation is also called the expected update as the equation is updated using the whole state space instead of a sample of the state space. One update is also called a sweep as the update sweeps through the state space. Once that the value function $v_{\pi}$ is updated, we know how good it is to follow the current policy. The next step is to deviate from the policy trajectory and choose a different action $a$ in state $s$ to find a more optimal policy value. We compute the new $\pi '$ and compare it to the old policy. The new policy is accepted if $\pi '(s) > \pi(s)$. This process is repeated until a convergence criterion is met {cite}`pashenkova1996value`. The complete algorithm can be found in the appendix. VI combines the policy evaluation with the policy improvement by truncating the sweep with one update of each state. It effectively combines the policy evaluation and policy evaluation in one sweep (see appendix for the algorithm) {cite}`pashenkova1996value` . PI and VI are the foundation of DP and numerous adaptions have been made to these algorithms. Although these algorithms do not have a wide application in many fields, their essential in describing what an RL algorithm effectively tries to approximate {cite}`sutton2018reinforcement`. 

# The Monte Carlo (MC) methods do not assume full knowledge of the dynamics of the environment and are thus considered model-free RL techniques. They only require a sample sequence of states, actions and rewards from the interaction of an environment. Technically, a model is still required which generates sample transitions, but the complete probability distribution $p$ of the dynamic system is not necessary. The idea behind almost all MC methods is that the agent learns the optimal policy by averaging the sample returns of a policy $\pi$ {cite}`arulkumaran2017brief`. They can therefore not learn on an online basis as after each episode they need to average their returns. Another difference between the two methods is that the MC method does not bootstrap like DP {cite}`mondal2020survey` . Meaning, each state has an independent estimate. Note that Monte Carlo methods create a nonstationary problem as each action taken at a state depends on the previous states. MC methods can either estimate $a$ state value {eq}`state-value` or estimate the value of a state-action pairs {eq}`value` (recall that the state-action values are the value of an action given a state). If state values are estimated, a model is required as it needs to be able to look ahead one step and choose the action which leads to the best reward and next state. With action value estimation you already estimated the value of the action and no model needs to be taken into account.  Monte Carlo methods also use a term called visits. A visit is when a state or state-action pair is in the sample path. Multiple visits to a state are possible in an episode. Two general Monte Carlo Methods can be deducted from visits. The every-visit MC methods and the first-visit MC methods. The every-visit MC methods estimate the value of a state as the average of the returns that have followed all visits to it. The first visit method only looks at the first visit of that state to estimate the average returns {cite}`van2012reinforcement`. The biggest hurdle in MC methods is that most state-action pairs might never be visited in the sample.
# 
# To overcome this problem multiple solutions have been explored. The naïve solution to this problem is called the exploring starts. Here, the idea is to allocate to each action in each state a nonzero probability at the start of the process. Although this is not possible in a practical setting where we truly want to interact with an environment, it enables us to improve the policy by making it greedy with respect to the current value function. As each state has a certain probability to explore, it will eventually explore the complete state space. If then an infinite number of episodes are taken, the policy improvement theory states that the policy $\pi$ will convergence to the optimal policy $\pi_*$ given the exploring starts {cite}`dolk2010survey`. Two other possibilities are applied in the field to solve this problem: on-policy methods and off-policy methods {cite}`sutton2018reinforcement`. On-policy methods attempt to improve on the current policy. This is also called a soft policy as $\pi(a|s) > 0$ for all $s \in S$ and all $ a \in A(s)$, but shifts eventually to the deterministic optimal policy. One of these on-policy methods is called an $\varepsilon$-greedy policy. The $\varepsilon$-greedy policy uses with probability $\varepsilon$ a random action instead of the greedy action. $\varepsilon$ is a fine-tuning parameter as it sets the balance between exploration and exploitation. The $\varepsilon$-soft policy is thus also a compromised solution as one cannot exploit and explore at the same rate. This is reelected by the fact that the $\varepsilon$-greedy policy is the best policy only among the $\varepsilon$-soft policies. The pseudocode of on-policy first visit MC for $\varepsilon$-soft policies algorithm can be found in the appendix. Lastly, the off-policy methods can be applied to overcome both the unrealistic exploring starts and the compromise needed in the on-policy methods. Off policy methods solve the exploration versus exploitation dilemma by considering two separate policies {cite}`van2012reinforcement`. one policy, called the target policy $\pi$, is being learned to become the optimal policy and another policy, called the behavior policy $b$, generates the behavior to explore the state space. In an off-policy method there needs to be coverage between the behavior policy and the target policy to transfer the exploration done by behavior policy $b$ to the target policy $\pi$. Meaning, every action taken under $\pi$ also needs to be taken occasionally under $b$. Consequently, the behavior policy needs to be stochastic in states where it deviates from the target policy. Complete coverage would imply that the behavior policy and the target policy are the same. The off-policy method would then become an on-policy method. The on-policy method can thus be viewed as a special case of off-policy in which the two policies are the same. Most off-policy methods use importance sampling to estimate expected values under one distribution given samples from another. Importance sampling uses the ratio of returns according to the relative probability of the trajectories of the target and behavior policies to learn the optimal policy {cite}`mahmood2014weighted`: 
# 
# $$ p_{t:T-1} = \frac{\prod^{T-1}_{k=t} \pi(A_k|S_k)p(S_{k+1}|S_k, A_k)}{\prod_{k=t}^{T-1}b(A_k|S_k)p(S_{k+1}|S_k,A_k)} = \prod_{k=t}^{T-1}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}$$
# 
# 
# 

# The formula is called the importance-sampling ratio. Note that the ratio only depends on the two policies and the sequence, not on the MDP. The importance-sampling ratio effectively transforms the expectations of $v_b(s)$ to have the right expectation.  Now, we can effectively estimate $v_{\pi}(s)$: 
# 
# $$V_{\pi}(s) = \frac{\sum_{t\in J(s)} p_{t:T-1}G_t}{|J(s)|}$$ 
# 
# Where $J(s)$ are all timesteps in which state $s$ is visited for an every-visit MC method and for a first-visit MC method $J(s)$ are all timesteps that were first visits to state $s$. An alternative to importance sampling is weighted importance sampling in which a weighted average is used: 
# 
# $$ V(s) = \frac{\sum_{t \in J(s)}p_{t:T-1}G_t}{\sum_{t \in J(s)}p_{t:T-1}} $$
# 
# The advantage of using a weighted importance sampling is a reduced variance as the variance is bounded when a weighting scheme is applied. The downside of this technique is that it increases the bias as the expectation deviates from the expectation of the target policy {cite}`mahmood2014weighted`.   

# The last general method to talk about is temporal-difference learning (TD). Temporal difference learning is a hybrid between Monte Carlo methods and Dynamic Programming. As DP, it updates estimates based on other learned estimates, not waiting on the final outcome, but it can learn directly from experience without a model of the environment like MC methods {cite}`sutton2018reinforcement`. The simplest TD method is the one-step TD. It updates the prediction of $v_{\pi}$ at each time step: 
# 
# $$V(S_t) \leftarrow V(S_t) + \alpha[R_{T+1} + \gamma V(S_{t+1}) - V(S_t)]$$
# 
# While MC method would update after each episode: 
# 
# $$ V(S_t) \leftarrow V(S_t) + \alpha[G_t - V(S_t)] $$
# 
# One-step TD effectively bootstraps the update like DP, but it uses a sampling estimate like the MC method to estimate V {cite}`rozonoer2018braverman`. The sampling estimate differs from the expected estimate on the fact that they are based on a single sample successor rather than on the complete distribution of all possible successors {cite}`li2016survey`. In the updating rule of TD methods there is the TD error (see quantity in brackets) which is the difference between the previous estimate of $S_t$ and the updated estimate  $R_{t+1} + \gamma V(S_{t+1} - V(S_t)$. The TD error is the error in the estimate made at that time. The psuedocode of the one-step TD method can be found in the appendix. TD methods lend themself quite easily to different methods in MC. For example, the Sarsa control algorithm is an on-policy TD in which the action values are updates using state-action pairs {cite}`li2016survey`: 
# 
# $$ q(s_t, a_t) \leftarrow q(s_t, a_t) + \alpha[R_{t+1} + \gamma q(s_{t+1}, a_{t+1}) - q(s_t, a_t)] $$
# 
# The same methodology is used here. $q_\pi$ is continuously estimated for policy $\pi$ while policy $\pi$ changes toward the optimal policy $\pi^*$ by a greedy approach. TD methods can also be applied to off-policy fashion. They are then called Q-learning which is widely applied in the literature. Q-learning is an off-policy method because they learn the action-value function $q$ independent of the policy being followed. They select the maximal or minimal action-value pair in the current state $s$: 
# ```{math}
# :label: Q-learning
# q(s_t, a_t) \leftarrow q(s_t, a_t) + \alpha[R_{t+1} + \gamma min_a q(s_{t+1}, a_{t+1}) - q(s_t, a_t)] 
# ```
# 
# The policy still has an effect in that it determines which states-action pairs are being visited, but the learned action-value function $q$ directly approximates $q_*$. This simplifies the analysis and enables early convergence. The last TD method is called the expected Sarsa and it uses the expected value instead of the minimum over the next state-action pairs to update the value function: 
# 
# $$  q(s_t, a_t) \leftarrow q(s_t, a_t) + \alpha[R_{t+1} + \gamma \mathbb{E}_{\pi}[q(s_{t+1}, a_{t+1})|S_{t+1}] - q(s_t, a_t)] $$
# 
# $$ q(s_t, a_t) \leftarrow q(s_t, a_t) + \alpha[R_{t+1} + \gamma \sum_a \pi(a|s_{t+1}) q(s_{t+1}, a) - q(s_t, a_t)] $$ 
# 
# The main benefit of Expected Sarsa over Sarsa is that it eliminates the variance caused by the random selection of $a_{t+1}$. Another benefit of Expected Sarsa is that it can be used as an off-policy method when the target policy $\pi$ is replaced with another policy {cite}`SanghiNimish2021DRLw`. 
# 
# These three methods lay at the foundation of RL and numerous adaptations have been made to fit the problem at hand. For example 
# 

# 
# ```{figure} C:/Users/ignac/Documents/GitHub/thesis/notebook/images/different_methods.png
# ---
# height: 350px
# width: 500px
# name: diff_meth-fig
# ---
# monte carlo temporal difference and dynammic programming
# ```

# ### Dimensions of a model-based reinforcement algorithm 

# {cite}`moerland2020framework` addresses the six most critical dimensions of an RL algorthim: computational effort, action value selection, cumulative return estimation, policy evaluation, function representation and update method. The first dimension has to do with the computational effort that is required to run the algorithm. The computational effort is primarily determined by the state set that is chosen (see {numref}`Figure {number} <state_space-fig>`). The first option is to consider all states $S$ of the dynamic environment. In practice, this often becomes impractical to consider due to the curse of dimensionality. The second and third possibilities are all reachable states and all relevant states. All reachable states are the states which are reachable from any start under any policy, while for the relevant states only those states under the optimal policy are considered. The last option is to use start states. These are all the states with a non-zero probability under $p(s_0)$  
# 
# (need examples and further explanaition curse of dimensionality)

# ```{figure} C:/Users/ignac/Documents/GitHub/thesis/notebook/images/state_space.png
# ---
# height: 350px
# width: 500px
# name: state_space-fig
# ---
# state_space dimensions  
# ```

# The second dimension is the action selection and has primarily to due to with exploration process of the algorithm. The first consideration in action selection is the candidate set that is considered for the next action. Then the optimal action needs to be considered while still keeping exploration in mind. For selecting the candidate set two main approaches are considered: step-wise and frontier. Frontier methods only start exploration once they are on the frontier, while step-wise methods have a new candidate set at each step of the trajectory. the MC method, DP and TD learning described above use step-wise exploration, while frontier methods are primarly used in robotics {cite}`8606991`. For the second consideration, selecting the action value,  different methods have been adopted. The first one is random explorations like $\varepsilon$-greedy exploration as explained in the section of Monte Carlo methods. These explorations techniques enable us to escape from a local minimum but can cause a jittering effect in which we undo an exploration step at random. The second approach is a value-based exploration that uses the value-based information to better direct the perturbation {cite}`yasuiempirical`. A good example of this are mean action values. They improve the random exploration by incorporating the mean estimates of all the available actions. Meaning, they explore actions with higher values more frequently than actions with lower values. The last option is state-based exploration. State-based exploration uses state-dependent properties to inject noise. Dynamic programming is a good example of this approach. DP is an ordered state-based exploration. Ordered state-based exploration sweeps through the state space in orded like tree structure. Other state-based explorations are possible like novelty and priors. 

# The dimensions of the calculation of the cumulative return estimation (see {eq}`return`) can be expressed in the formula to address the practical issues and limitations in RL: 
# 
# $$ G_t = \sum_{k=0}^T\gamma^kR_{t+k+1} $$ 
# 
# $$ q(s,a) = E[G_t| S_t = d, A_t = a]$$
# 
# $$ \hat{q}(s,a) = \sum_{k=0}^T \gamma^kR_{t+k+1} + \gamma^KB(s_{t+T}) $$ 
# 
# Where $T \in {1,2,3, ..., \infty}$ denotes the sample depth and $B(.)$ is a bootstrap function. For the sample depth three possible option are possible: $K = \infty$, $K = 1$,  $K = n$ or reweighted. Monte Carlo methods for example use a sample depth to infinity as they do not bootstrap at all. Instead, DP uses bootstrapping at each iteration, so $K = 1$. An intermediate method between DP and Monte Carlo methods can also be devised in which $ K = n$. The reweighted option is a special case of $ K = n$ in which targets of different depths are combined with a weighting scheme. The bootstrap function can be devised using a learned value function like the state value function or the state-action value function or following a heuristic approach. A good heuristic can be obtained by first solving a simplified version of the problem. An example of this is first solving the deterministic problem and then using the solution as a heuristic on its stochastic counterpart {cite}`moerland2020framework`. The second dimension in cumulative return estimation is whether full knowledge of the dynamic system is in place (full backups) or a sample is taken from the environment (sample backups) {cite}`arulkumaran2017brief`. In {numref}`Figure {number} <backups-bootstrap-fig>` these two dimensions are represented. 
#  
# 
# 

# ```{figure} C:/Users/ignac/Documents/GitHub/thesis/notebook/images/backups.png
# ---
# height: 250px
# width: 300px
# name: backups-bootstrap-fig
# ---
# consideration in calculating the cumulative return estimation   
# ```

# The fourth dimension to consider is policy evaluation. Policy evaluation has two dimensions. One is on which policy to use: on-policy or off-policy method. We have already seen this dimension in the section of MC methods and it will not be further discussed. Another dimension is function representation. The first choice that needs to be made here is which function to represent. In theory, we have two essential functions: the value function and the policy function. The value function can be the state-action value function or just the state value function, but primarily represents the value of the current or optimal policy at all considered state-action pairs. The policy function on the other hand maps every state to a probability distribution over actions and is best used in continuous action spaces as we can directly act in the environment by sampling from the policy distribution {cite}`moerland2020framework`. The second choice is how to represent this function. There are two possibilities here. The first option is using a tabular approach in which each state is a unique element for which we store an individual estimate.  This can be done on a global level or local level. At the global level, the entire state space is encapsulated by the table. Unfortunately, this method does not scale well and is only appliable in small exploratory problems. On the contrary, a local table does scale well as it is built temporarily until the next real step. The other method for function representation is function approximation. Function approximation builds on the concept of generalization. Generalization assumes that similar states to function will in general also have approximately similar output predictions (Generalization is further discussed in next section).  Function approximation uses this to share information between near similar states and therefore store a global solution for a larger state space {cite}van2012reinforcement. There are two kinds of function approximations: parametric and non-parametric. A good example of a parametric function approximation is a neural network and for non-parametric a k-nearest neighbors can be thought of. The big challenge in function approximation is finding the balance between overfitting and underfitting the actual data.
# 
# The last dimension is the updating method. The updating method used should be in line with the function representation and the policy evaluation method as certain updating rules only work on a set of function representation and policy evaluation methods {cite}`moerland2020framework`. For the updating method, there are quite a few choices to make. The first choice is choosing between gradient-based updates and gradient-free updates. In gradient-based updates we repeatedly update our parameters in the direction of the negative gradient loss with respect to the parameters:
# 
# $$ \theta \leftarrow \theta - \alpha \cdot \frac{\partial L(\theta)}{\partial \theta} $$ 
# 
# Where $\alpha \in \mathbb{R}^+$ is a learning rate. Before the updating rule can be applied a loss function $L(\theta)$ should first be chosen. The loss function is usually a function of both the function representation and the policy evaluation method. As there are two kinds of function to represent in function representation, there are also two kinds of losses: value loss and policy loss. The most general value loss is the mean squared error loss. In policy loss, there are various methods to estimating the loss. For example the policy gradient specifies a relation between the value estimates $\hat{q}(s_t,a_t)$ and the policy $\pi_{\theta}(a_t|s_t)$ by ensuring that actions with high values also get high policy probabilities assigned: 
# 
# $$L(\theta|s_t, a_t) = -\hat{q}(s_t,a_t) \cdot ln (\pi_{\theta}(a_t|s_t))$$
# 
# Once the loss function is defined, the gradient-based updating rule can be applied. The updating again depends on the function representation for example the value update on a table for the mean squared loss function becomes: 
# 
# $$ q(s,a) \leftarrow q(s,a) - \alpha \cdot \frac{\partial L(q(s,a))}{\partial q(s,a)} $$
# $$ \frac{\partial L(q(s,a))}{\partial q(s,a)} = 2 \cdot \frac{1}{2}(q(s,a) - \hat{q}(s,a)) $$
# 
# $$ q(s,a) \leftarrow q(s,a) - \alpha(q(s,a) - \hat{q}(s,a)) $$ 
# $$ q(s,a) \leftarrow (1- \alpha) \cdot q(s,a) + \alpha \cdot \hat{q}(s,a)  $$
# 
# Where q(s,a) is a table entry. The same can be done for function approximation where the derivative of the loss function then becomes: 
# 
# $$ \frac{\partial L(\theta)}{\partial \theta} = (q(s,a) - \hat{q}(s,a)) \cdot \frac{\partial q(s,a)}{\partial \theta} $$
# 
# Where $\frac{\partial q(s,a)}{\partial \theta}$ can be for example the derivatives in a neural newtork. 
# 
# Gradient-free updating rules use a parametrized policy function and then repeatedly perturb the parameters in policy space, evaluate the new solution by sampling traces and decide whether the perturbed solution should be retained. they only require an evaluation function and treat the problem as a black-box optimization setting. Gradient-free updating methods are thus not fit for model-based RL. An overview of the different dimensions can be viewed in {numref}`Figure {number} <dim-fig>`. 
# 

# ```{figure} C:/Users/ignac/Documents/GitHub/thesis/notebook/images/dimensions.png
# ---
# height: 250px
# width: 600px
# name: dim-fig
# ---
#  The dimensions of a reinforcment learning algorithm   
# ```

# ## Curse of Dimensionality and Model-based Deep Reinforcment Learning

# For continuous states and actions, which is the most relevant case for optimal control. The state and action dimension are infinite. This require function approximation methods to estimate the optimal value function $v_{\pi^*}$ {cite}`shin2019reinforcement`. The most notable function approximations are neural networks. Especially deep neral networks (DNN) can significantly reduce the time and effort required to approximate the value function {cite}`shin2019reinforcement`. The use of parrallelization to speed up and stabilize the learning process {cite}`shin2019reinforcement`. 

# ## G-learning, a stochastic adaptation on Q-learning 

# Q-learning learns extremely slow in noisy environments due to the minimization bias. In {eq}`Q-learning` the minimum over the estimated values is used implicitly as an estimate of the minimum value, which can lead to significant positive bias in noisy environments. Consider, for example, a single state $s$ where there are many actions $a$ whose true values are all zero but whose estimated values are uncertain and thus distributed some above and some below zero. The minimum of the true values is zero, but the minimum of the estimates is negative. Consequently, introducing minimization bias {cite}´sutton2018reinforcement´. This can also be illustrated by the Jensen's inequality for the concave min operator. Assume that $Q(s,a)$ is an unbiased but noisy estimate of the optimal $Q^*(s,a)$. Then it applieas that 
# 
# $$ \mathbb{E}[min_aQ(s,a)] \leq min_aQ^*(s,a)$$
# 
# This creates an optimistic bias, causing the cost-to-go to appear lower than it is. The minimization bias has an impact on the learning rate of the Q-learning policy. The impact depends on the gap $Q^*(s,a') - V^*(s)$  between the value of a non-optimal action $a'$ and that of the optimal action. If the gap is large, $a'$ seems suboptimal as desired. If the gap is small, confusing $a'$ for the optimal action does not effect the learning process. However, when the gap is in the order of the noise term, the minimization bias has a significant impact, because $a'$ does not appear to be suboptial and is thus still accepted as optimal. The optimstic bias is futher enhanced by propegating the bias between states and can lead to regions of the state space that are highly biased creating large-gap suboptimal actions. Although this problem hampers the learning rate, Q-learning can still learn in a stochstic environment due to the fact that the bias draws exploration towards the given state, leading to a decrease in variance, which in turn reduces the bias {cite}`fox2015taming`. 
# 
# G-learning is an adaptation of Q-learning, specifically designed to handle noisy environments. It is also an off-policy approach in a model-free setting, but it regularizes the state-action value function learned by an agent. It regularizes the state-action value function by penalizing deterministic policies early in the optimization process. Penilazation is done early in the process, because there is still a small sample size and therefore a more randomized policy is prefered. When the sample size grows, one should expect to shifts to a more deterministic and exploiting policy. This is what G-learning effectively does. It adds a cost-to-go term to the value function that penalizes the early deterministic policies which diverge from a simple stochastic prior policy $\rho(a|s)$. The prior stochastic policy sets up an information cost of a learned policy $\pi(a|s)$, effectively penalizing deviations from the prior policy . 
# 
# $$ g^{\pi}(s,a) = log(\frac{\pi(a,s)}{\rho(a,s)}) $$
# 
# taken the expectation of the policy $\pi$ gives us the Kullback-Leibler divergence of the two policies. 
# 
# $$ \mathbb{E}_{\pi}[g^{\pi}(s,a)|s] = D_{KL}[\pi_s || \rho_s] $$ 
# 
# Now consider the total discounted expected information cost 
# 
# ```{math}
# :label: information cost
# I^{\pi}(s) = \sum_{t\geq0} \gamma^t \mathbb{E}[g^{\pi}(s_t,a_t)|s_0=s]
# ```
# 
# Adding {eq}`information cost` to the value function {eq}`value` gives the free-energey function 
# 
# 
# $$F^{\pi}(s)= V^{\pi}(s) + \frac{1}{\beta} I^{\pi}(s)$$
# ```{math}
# :label: free-energy
# \hspace{3cm} = \sum_{t\geq 0 } \gamma^t \mathbb{E}[\frac{1}{\beta} g^{\pi}(s_t,a_t) + R_t|s_0=s] 
# ```
# 
# Where $\beta$ is the parameter which sets the weight of the information cost in the value function. If $\beta$ is small, $\pi$ will act similar to $\rho$. When $\beta$ is large, $\pi$ will diverge from the prior and will therefore approache the greedy policy of Q-learning. A smooth transition between small and large values for \beta will allow the algorithm to avoid early deterministic policies and still be able to exploit the optimal values. The same approach can be done for the state-action value function $q(s,a)$
# 
# ```{math}
# :label: free-q
# H^{\pi}(s,a) = \sum_{t\geq 0 } \gamma^t \mathbb{E}[R_t + \frac{\gamma}{\beta} g^{\pi}(s_{t+1}, a_{t+1})|s_0=s, a_0=a] 
# ```
# 
# Notice that the information term at time $t=0$ is not needed as the action $a_0 =a$ is already known. Given {eq}`free-energy` and {eq}`free-q` it follows that
# 
# 
# ```{math}
# :label: free-energy2
# F^{\pi}(s) = \sum_a \pi(a|s)[\frac{1}{\beta}log\frac{\pi(a|s)}{\rho(a|s)} + H^{\pi}(s,a)]
# ```
# 
# The gradient of $F^{\pi}$ at zero is 
# 
# 
# ```{math}
# :label: soft-min
# \pi(a|s) = \frac{\rho(a|s)e^{-\beta H(s,a)}}{\sum_{a'} \rho(a'|s)e^{-\beta H(s,a')}}
# ```
# 
# {eq}`soft-min` is the soft-min operator applied to H. Now evaluate {eq}`free-energy2` at {eq}`soft-min` 
# 
# $$ F^{\pi}(s) = \frac{-1}{\beta} log(\sum_a \rho(a,s) e^{-\beta H^{\pi}(s,a)}) $$
# 
# This expression can get pluged in {eq}`free-q`and as a result the optimal $H^*$ is achieved. 
# 
# $$ H^*(s,a) = \mathbb{E}[R|s,a] -\frac{\gamma}{\beta} \mathbb{E}[log \sum_{a'} \rho(a'|s')e^{-\beta H^*(s',a')}] $$
# 
# Given the above expression, the Q-learner is adapted to learn the optimal G^* from interaction with the environment by applying the update rule 
# 
# $$ G(s_t, a_t) \leftarrow (1-\alpha_t)G(s_t,a_t) + \alpha_t(c_t - \frac{\gamma}{\beta}log(\sum_{a'}e^{-\beta H(s_{t+1}, a')})) $$

# The G-learner is applied by to 
# 
# in the next subsection we will describe the general approach of the Deep BSDE and link it to the RL theory. Although this method is orginaly designed to only solve the equation at timestep $t=0$, future research might be able to solve the PDE at each time point $t$. 

# ## The Deep Backward Stochastic Differential Equation Method

# The general PDEs that Deep BSDE method solves can be written as: 
# 
# ```{math}
# :label: gen_form
# \frac{\partial u}{\partial t} + \frac{1}{2} Tr(\sigma \sigma^T (Hess_xu) + \Delta u(t,x)  \mu(t,x) + f(t,x,u, \sigma^T(\Delta_x u)) = 0 
# ``` 
# with some temrinal condition $u(T,x) = g(x)$. 
# 
# With $u(T,x) = L(x)$. The key idea is to  reformulate the PDE as an appropriate stochastic problem {cite}`weinan2020algorithms` and {cite}`weinan2017deep`. Here the probability space ($\Omega,\mathcal{F}, \mathbb{P}$) is adapted to the high dimensional problem. So $W: [0, T] \times \Omega \rightarrow \mathbb{R}^d$ becomes a d-dimensional standard Brownian motion on ($\Omega,\mathcal{F}, \mathbb{P}$) and let $\mathcal{A}$ be the set of all $\mathbb{F}$-adapted $\mathcal{R^d}$-values stochastic processes with continuous sample paths. Let $\{X_T\}_{0 \leq t \leq T}$ be a d-dimensional stochastic process which satisfies
# 
# $$ X_t = \varepsilon + \int_0^t \mu(s,X_s)ds + \int_0^t \sigma(s,X_s)dW_s $$
# 
# Using Itô's lemma, we obtain that 
# 
# $$ y(t, X_t) - u(0,X_0) = - \int_0^t f(s,X_s,u(s,X_s), [\sigma(s,X_s)]^T(\Delta_x u)(s,X_s)) ds + \int_0^t[\Delta u(s,X_s)]^T\sigma(s,X_s)dW_s$$ 
# 
# A backward stochstic differential equation can be written as 
# ```{math}
# :label: stoch_con
# \begin{cases} X_t = \varepsilon + \int_0^t \mu(s,X_s) ds + \int_0^t\sigma(s,X_S)dW_S \\ 
# Y_t = g(X_T) + \int_t^T f(s, X_s, Y_s, Z_s)ds - \int_t^T(Z_s)^T dW_s
# \end{cases} 
# ```
# 
# In the literature it was found that the solution of PDE and its spatial derivative are now the solution of the stochastic control problem {eq}`stoch_con` {cite}`weinan2020algorithms`. The relationship between the PDE {eq}`gen_form` and the BSDe {eq}`stoch_con` is based on the nonlinear Feynman-Kac formula {cite}`bloch2018machine` and {cite}`guler2019towards`. Under suitable additional regularity assumption on the nonlinearity $f$ in the sense that for all $t \in[0,T]$ it holds $\mathbb{P}$-a.s. that 
# 
# ```{math}
# :label: identity
# Y_t = u(t, \epsilon + W_t) \in \mathbb{R}  \hspace{0.2cm}\text{and}\hspace{0.2cm} Z_t = (\Delta_x u) (t, \epsilon + W_t) \in \mathbb{R}^d
# ```
# 
# The first identity in {eq}`identity` is referred to as nonlinear Feynman-Kac formula {cite}`weinan2017deep`. $(Y_t, Z_t), t \in [0,T]$ is a solution for the BSDE and with {eq}`identity` in mind the PDE problem can be formulated as the following variational problem: 
# 
# $$ inf_{Y_0,\{Z_T\}_{0\leq t\leq T}} \mathbb{E}[|g(X_T) - Y_T|^2] $$
# $$ s.t. \hspace{0.2cm}X_T = \varepsilon + \int_0^t \mu(s,X_s)ds + \int_0^t\sum(s,X_s)dW_s $$ 
# $$  \hspace{1.2cm}Y_t = Y_0 - \int_0^th(s,X_s,Y_s,Z_s)ds + \int_0^t(Z_s)^TdW_s$$ 
# 
# The minimizer of this variational problem is the solution to the PDE {cite}`raissi2018forward`. The main idea behind Deep BSDE method is to approximate the unknown function $X_0 \rightarrow u(, X_0)$ and $X_t \rightarrow [\sigma(t,X_t)]^T((\Delta_x u)(t,X_t)$ by two feedforward neural networks $\psi$ and $\phi$ {cite}`han2018solving`. To achieve this we discretize time using Euler scheme on a grid $ 0 = t_0<t_1<...<T_N =T $
# 
# $$ inf_{\psi_0,\{\phi_n\}^{N-1}_{n=0}} \mathbb{E}[|g(X_T) - Y_T|^2] $$
# $$ s.t. \hspace{0.2cm} X_0 = \varepsilon, \hspace{0.2cm} Y_0 = \psi_0(\varepsilon)$$
# $$ \hspace{3.2cm} X_{t_{n+1}} = X_{t_i} \mu(t_n,X_{t_n}) \Delta t + \sigma(t_n,X_{t_n}) \Delta W_n$$ 
# $$ Z_{t_n} = \psi_(X_{t_n}) \hspace{0.4cm}$$
# $$  \hspace{1.2cm}Y_{t_{n+1}} = Y_{t_n} - f(t_n,X_{t_n},Y_{t_n},Z_{t_n})\Delta t  + (Z_{t_n})^T \Delta W_n$$ 
# 
# At each time slide $t_n$, a subnetwork is associated. These subnetworks are then stacked together to form a deep composite neural network {cite}`han2017overcoming`. The network takes the paths  $\{X_{t_n}\}_{0\leq n \leq N}$ and $\{W_{t_n}\}_{0\leq n \leq N}$ as the input data and gives as final output, denoted by $\hat{u}(\{ X_{t_n}\}_{0 \leq n \leq N}, \{W_{t_n}\}_{0 \leq n \leq N}$,  as an approximation to $u(t_N, X_{t_N})$ (see {numref}`Figure {number} <BSDN-fig>`) {cite}`han2018solving`. Thereby it is only solved a time step $t=0$. The difference in the matching of a given terminal condition can be used to define the expected loss function {cite}`weinan2020algorithms` {cite}`han2017overcoming`
# 
# $$ m(\theta) = \mathbb{E}[|g(X_{t_N}) - \hat{u}(\{ X_{t_n}\}_{0 \leq n \leq N}, \{W_{t_n}\}_{0 \leq n \leq N} |^2]$$ 
# 
# 

# ```{figure} C:/Users/ignac/Documents/GitHub/thesis/notebook/images/BSDE_NN.png
# ---
# height: 300px
# width: 500px
# name: BSDN-fig
# ---
# neural network for Deep BSDE method
# ```

# An other way to look at it is that the stochastic control problem is a model-based reinforcement learning problem {cite}`han2018solving`. In this setting $Z$ is viewed as the policy we try to approximate using a feedforward neural network. The process $u(t, \varepsilon + W_t), t \in [0, T]$, corresponds to the value function associated with the stochastic control problem and can be approximately employed by the policy Z {cite}`weinan2017deep`. A benefit of using deep BSDE method is does not require us to generate training data beforehand. The paths play the role of the data and they are generated on the spot {cite}`weinan2020algorithms`. 
# 
# The deep BSDE method solves the PDE for $Y_0= u(0, X_0) = u(0, \varepsilon)$. This means that in order to obtain an approximate of $Y_t = u(t,X_t)$ at a later time $t>0$, we will have to retain our algorithm. {cite}`raissi2018forward` solves this isue by directly placing a neural network on the object of interest, the unknown solution $u(t,x)$
