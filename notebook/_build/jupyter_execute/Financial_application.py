#!/usr/bin/env python
# coding: utf-8

# # Financial Application of Reinforcement Learning

# Advancements in Reinforcement Learning can be applied to a financial setting. In this subsection, a model is proposed for optimal consumption, life insurance and investment. This model can be extended in multiple ways and can be applied in a broad range of financial scenario's. The ony problem of the model is the curse of dimensionality which makes the Hamilton-Jacobi-Bellman (HJB) equation of the model in higher dimensions impossible to solve. A deep learning-based approach that can handle the HJB equation in higher dimensions is proposed. Although the method originates from optimal control literature, it resembles many feutures of the reinforcement algorithms and it solves the exact same problem.  

# ## Optimal consumption, investment and life insurance in an intertemporal model

# The first person to include uncertain lifetime and life insurance decisions in a discrete life-cycle model was Yaari {cite}`yaari1965uncertain`. He explored the model using a utility function without bequest (Fisher Utility function) and a utility function with bequest (Marshall Utility function) in a bounded lifetime. In both cases, he looked at the implications of including life insurance. Although Yaari's model was revolutionary in the sense that now the uncertainty of life could be modeled, Leung {cite}`leung1994uncertain` found that the constraints laid upon the Fisher utility function were not adequate and lead to terminal wealth depletion. Richard {cite}`richard1975optimal` applied the methodology of Merton {cite}`merton1969lifetime, merton1975optimum` to the problem setting of Yaari in a continuous time frame. Unfortunately, Richard's model had one deficiency: The bounded lifetime is incompatible with the dynamic programming approach used in Merton's model. As an individual approaches his maximal possible lifetime T, he will be inclined to buy an infinite amount of life insurance. To circumvent this Richard used an artificial condition on the terminal value. But due to the recursive nature of dynamic programming, modifying the last value would imply modifying the whole result. Ye {cite}`ye2006optimal`  found a solution to the problem by abandoning the bounded random lifetime and replacing it with a random variable taking values in $[0,\infty)$. The models that replaced the bounded lifetime, are thereafter called intertemporal models as the models did not consider the whole lifetime of an individual but rather looked at the planning horizon of the consumer.  Note that the general setting of Ye {cite}`ye2006optimal` has a wide range of theoretical variables, while still upholding a flexible approach to different financial settings. On this account, it is a good baseline to confront the issues concerning the current models of financial planning. However, one of the downsides of the model is the abstract representation of the consumer. Namely, the rational consumer is studied, instead of the actual consumer. To detach the model from the notion of rational consumer, I will more closely look at behavioral concepts that can be implemented. In the next paragraph various modification will be discussed and a further review is conducted on the behavioral modifications
# 
# 
# After Ye {cite}`ye2006optimal` various models have been proposed which all have given rise to unique solutions to the consumption, investment, and insurance problem. The first unique setting is a model with multiple agents involved. For example,  Bruhn and Steffensen {cite}`bruhn2011household` analyzed the optimization problem for couples with correlated lifetimes with their partner nominated as their beneficiary using a copula and common-shock model, while Wei et al.{cite}`wei2020optimal` studied optimization strategies for a household with economically and probabilistically dependent persons. Another setting is where certain constraints are used to better describe the financial situation of consumers. Namely, Kronborg and Steffensen {cite}`kronborg2015optimal` discussed two constraints. One constraint is a capital constraint on the savings in which savings cannot drop below zero. The other constrain involves a minimum return in savings. A third setting describes models who analyze the financial market and insurance market in a pragmatic environment. A good illustration is the study of Shen and Wei {cite}`shen2016optimal`. They incorporate all stochastic processes involved in the investment and insurance market where all randomness is described by a Brownian motion filtration. An interesting body of models is involved in time-inconsistent preferences. In this framework, consumers do not have a time-consistent rate of preference as assumed in the economic literature. There exists rather a divergence between earlier intentions and later choices De-Paz et al. {cite}`de2014consumption`. This concept is predominantly described in psychology. Specifically, rewards presented closer to the present are discounted proportionally less than rewards further into the future. An application of time-inconsistent preferences in the consumption, investment, and insurance optimization can be found in Chen and Li {cite}`chen2020time` and De-Paz et al. {cite}`de2014consumption`. These time-inconsistent preferences are rooted in a much deeper behavioral concept called future self-continuity. Future self-continuity can be described as how someone sees himself in the future. In classical economic theory, we assume that the degree to which you identify with yourself has no impact on the ultimate result. In the next subsection, the relationship of future self-continuity and time-inconsistent preferences are more closely looked at and future self-continuity is further examined in the behavioral life-cycle model. 

# ### The model specifications

# In this section, I will set the dynamics for the baseline model in place. The dynamics follow primarily from the paper of Ye {cite}`ye2006optimal`.
# 
# Let the state of the economy be represented by a standard Brownian motion $w(t)$, the state of the consumer's wealth be characterized by a finite state multi-dimensional continuous-time Markov chain $X(t)$ and let the time of death be defined by a non-negative random variable $\tau$. All are defined on a given probability space ($\Omega, \mathcal{F}, \mathbb{P} $) and $W(t)$ is independent of $\tau$. Let $T< \infty$ be a fixed planning horizon. This can be seen as the end of the working life for the consumer. $\mathbb{F} = \{\mathcal{F}_t, t \in [0,T]\}$, be the P-augmentation of the filtration $\sigma${$W(s), s<t \}, \forall t \in [0,T]$ , so $\mathcal{F}_t$ represents the information at time t. The economy consist of a financial market and an insurance market. In the following section I will construct these markets separetly following Ye {cite}`ye2006optimal`. 

# The financial market consist of a risk-free security $B(t)$ and a risky security $S(t)$, who evolve according to 
# 
# $$ \frac{dB(t)}{B(t)}=r(t)dt $$
# 
# $$ \frac{dS(t)}{S(t)}=\mu(t)dt+\sigma(t)dW(t)$$
# 
# Where $\mu, \sigma, r > 0$ are constants and $\mu(t), r(t), \sigma(t): [0,T] \to R$ are continous. With $\sigma(t)$ satisfying $\sigma^2(t) \ge k, \forall t \in [0,T]$

# The random variable $\tau_d$ needs to be first modeled for the insurance  market. Assume that $\tau$ has a probability density function $f(t)$ and probability distribution function given by 
# 
# $$ F(t) \triangleq P(\tau < t) = \int_0^t f(u) du $$
# 
# Assuming $\tau$ is independent of the filtration $\mathbb{F}$ 
# 
# Following on the probability distribution function we can define the survival function as followed
# 
# $$ \bar{F}(t)\triangleq P(\tau \ge t) = 1 -F(t) $$
# 
# The hazard function is the  instantaneous death rate for the consumer at time t and is defined by 
# 
# $$ \lambda(t) = \lim_{\Delta t\to 0} = \frac{P(t\le\tau < t+\Delta t| \tau \ge t)}{\Delta t} $$
# 
# where $\lambda(t): [0,\infty[ \to R^+$ is a continuous, deterministic function with $\int_0^\infty \lambda(t) dt = \infty$.
# 
# Subsequently, the survival and probability density function can be characterized by 
# 
# 
# $$ \bar{F}(t)= {}_tp_0= e^{-\int_0^t \lambda(u)du} $$
# $$ f(t)=\lambda(t) e^{-\int_0^t\lambda(u)du} $$ 
# 
# With conditional probability described as 
# 
# $$ f(s,t) \triangleq \frac{f(s)}{\bar{F}(t)}=\lambda(s) e^{-\int_t^s\lambda(u)dy} $$
# $$ \bar{F}(s,t) = {}_sp_t \triangleq \frac{\bar{F}(s)}{\bar{F}(t)} = e^{-\int_t^s \lambda(u)du} $$
# 
#     
# Now that $\tau$ has been modeled, the life insurance market can be constructed. Let's assume that the life insurance is continuously offered and that it provides coverage for an infinitesimally small period of time. In return, the consumer pays a premium rate p when he enters into a life insurance contract, so that he might insure his future income. In compensation he will receive  a total benefit of $\frac{p}{\eta(t)}$ when he dies at time t. Where $\eta : [0,T] \to R^+ $ is a continuous, deterministic function.
# 
# Both markets are now described and the wealth process $X(t)$ of the consumer can now be constructed. Given an initial wealth $x_0$, the consumer receives a certain amount of income $i(t)$ $\forall t \in [0,\tau \wedge T]$ and satisfying $\int_0^{\tau \wedge T} i(u)du < \infty$. He needs to choose at time t a certain premium rate $p(t)$, a certain consumption rate $c(t)$ and a certain amount of his wealth $\theta (t)$ that he invest into the risky asset $S(t)$. So given the processes $\theta$, c, p and i, there is a wealth process $X(t)$  $\forall t \in [0, \tau \wedge T] $ determined by 
# 
# $$ dX(t) = r(t)X(t) + \theta(t)[( \mu(t) - r(t))dt +\sigma(t)dW(t)] -c(t)dt -p(t)dt + i(t)dt,   \quad t \in [0,\tau \wedge T] $$
# 
# If $t=\tau$ then the consumer will receive the insured amount $\frac{p(t)}{\eta(t)}$. Given is wealth X(t) at time t his total legacy will be 
# 
# $$ Z(t) = X(t) + \frac{p(t)}{\eta(t)} $$ 
# 
# The predicament for the consumer is that he needs to chose the optimal rates for c, p , $\theta$ from the set $\mathcal{A}$ , called the set of admissible strategies, defined by 
# 
# $$ \mathcal{A}(x) \triangleq  \textrm{set of all possible triplets (c,p,}\theta) $$ 
# 
# such that his expected utility from consumption, from legacy when $\tau > T$ and from terminal wealth when $\tau \leq T $  is maximized. 
# 
# $$ V(x) \triangleq \sup_{(c,p,\theta) \in \mathcal{A}(x)} E\left[\int_0^{T \wedge \tau} U(c(s),s)ds + B(Z(\tau),\tau)1_{\{\tau \ge T\}} + L(X(T))1_{\{\tau>T\}}\right] $$ 
# 
# Where $U(c,t)$ is the utility function of consumption, $B(Z,t)$ is the utility function of legacy and $L(X)$ is the utility function for the terminal wealth. $V(x)$ is called the value function and the consumers wants to maximize his value function by choosing the optimal set $\mathcal{A} = (c,p,\theta)$. The optimal set $\mathcal{A}$ is found by using the dynamic programming technique described in the following section. 

# ### dynamic programming principle 

# To solve the consumer's problem the value function needs to be restated in a dynamic programming form. 
# 
# $$J(t, x; c, p, \theta) \triangleq E \left[\int_0^{T \wedge \tau} U(c(s),s)ds + B(Z(\tau),\tau)1_{\{\tau \ge T\}} + L(X(T))1_{\{\tau>T\}}| \tau> t, \mathcal{F}_t \right] $$
# 
# The value function becomes
# 
# $$ V(t,x) \triangleq \sup_{\{c,p,\theta\} \in \mathcal{A}(t,x)} J(t, x; c, p, \theta)  $$
# 
# Because $\tau$ is independent of the filtration, the value function can be rewritten as 
# 
# $$ E \left[\int_0^T  \bar{F}(s,t)U(c(s),s) + f(s,t)B(Z(\tau),\tau) ds  + \bar{F}(T,t)L(X(T))| \mathcal{F}_t \right]$$ 
# 
# The optimization problem is now converted from a random  closing time point to a fixed closing time point. The mortality rate can also be seen as a discounting function for the consumer as he would value the utility on the probability of survival. 
# 
# Following the dynamic programming principle we can rewrite this equation as the value function at time s plus the value created from time step t to time step s. This enables us to view the optimization problem into a time step setting, giving us the incremental value gained at each point in time.   
# 
# $$ V(t,x) = \sup_{\{c,p,\theta\} \in \mathcal{A}(t,x)} E\left[e^{-\int_t^s\lambda(v)dv}V(s,X(s)) + \int_t^s f(s,t)B(Z(s),s) + \bar{F}(s,t)U(c(s),s)ds|\mathcal{F}_t\right] $$ 
# 
# The Hamiltonian-Jacobi-bellman (HJB) equation can be derived from the dynamic programming principle and is as follows
# 
# ```{math}
# :label: BELL
# 
# \begin{cases} 
# V_t(t,x) -\lambda V(t,x) + \sup_{(c,p,\theta)} \Psi(t,x;c,p,\theta)  = 0 \\ V(T,x) = L(x)  
# \end{cases}
# 
# ```
# 
# where 
# 
# $$ \Psi(t,x; c,p,\theta) \triangleq r(t)x + \theta(\mu(t) -r(t)) + i(t) -c -p)V_x(t,x) + \\ \frac{1}{2}\sigma^2(t)\theta^2V_{xx}(t,x) + \lambda(t)B(x+ p/\eta(t),t) + U(c,t) $$
# 
# 
# Proofs for deriving the HJB equation, dynammic programming principle and converting from a random closing time point to a fixed closing time point can be found in Ye {cite}`ye2006optimal`
# 
# A strategy is optimal if  
# 
# 
# \begin{gather*}
# 0 =V_t(t,x) -\lambda(t)V(t,x) + \sup_{c,p,\theta}(t,x;c,p,\theta)  \\
# 0 = V_t(t,x) -\lambda(t)V(t,x) + (r(t)x+ i(t))V_x + \sup_c\{U(c,t)-cV_x\} + \\ \sup_p\{\lambda(t)B(x + p/\eta(t),t) - pV_x\} + \sup_\theta \{ \frac{1}{2}\sigma^2(t)V_{xx}(t,x)\theta^2 +(\mu(t) - r(t))V_x(t,x)\theta\} 
# \end{gather*}
# 
# 
# The first order conditions for regular interior maximum are
# ```{math}
# :label: cons_cond
# \sup_c  \{ U(c,t) - cV_x\} = \Psi_c(t,x;c^*,p^*,\theta^*)  \rightarrow  0 = -V_x(t,x) + U_c(c*,t) 
# ``` 
# 
# ```{math}
# :label: ins_cond
# \sup_p\{\lambda(t)B(x + p/\eta(t),t) - pV_x\} = \Psi_p(t,x;c^*,p^*,\theta^*) \\ \rightarrow 0 = -V_x(t,x) + \frac{\lambda(t)}{\eta{t}}B_Z(x + p^*/\eta(t),t)
# ```
# 
# ```{math}
# :label: inv_cond
# \sup_\theta \{ \frac{1}{2}\sigma^2(t)V_{xx}(t,x)\theta^2 +(\mu(t) - r(t))V_x(t,x)\theta\} = \Psi_\theta(t,x;c^*,p^*,\theta^*)\\ \rightarrow 0 = (\mu(t) -r(t))V_x(t,x) + \sigma^2(t)\theta^*V_{xx}(t,x)
# ```
# 
# The second order conditions are 
# 
# $$ \Psi_{cc}, \Psi_{pp}, \Psi_{\theta \theta} < 0 $$ 
# 

# This optimal control problem has been solved analytically by Ye {cite}`ye2006optimal` for the Constant Relative Risk Aversion utility function. In the next subsection the numerical method to solve the problem in higher dimensions is introduced. To solve {eq}`BELL` a new technique called the Deep Backward Stochastic Differential Equation (BSDE) method can be used. The Deep BSDE method was the first deep learning-based numerical algorithm to solve general nonlinear parabolic PDEs in high dimensions. in the next subsection we will describe the general approach of the Deep BSDE and link it to the RL theory. Although this method is orginaly designed to only solve the equation at timestep $t=0$, future research might be able to solve the PDE at each time point $t$. 

# ## Numerical method: The Deep Backward Stochastic Differential Equation method

#  The general PDEs that this method can solve can be written as: 

# ```{math}
# :label: gen_form
# \frac{\partial u}{\partial t} + \frac{1}{2} Tr(\sigma \sigma^T (Hess_xu) + \Delta u(t,x)  \mu(t,x) + f(t,x,u, \sigma^T(\Delta_x u)) = 0 
# ``` 
# with some temrinal condition $u(T,x) = g(x)$. {eq}`BELL` can thus be reformulated in the general form: 
# 
# $$\underbrace{V_t(t,x)}_{\frac{\partial u}{\partial t}} + \underbrace{\frac{1}{2}\sigma(t)^2\theta^2V_{xx}(t,x)}_{\frac{1}{2}Tr(\sigma \sigma^T(Hess_xu(t,x)))} + \underbrace{(r(t) x + \theta(\mu(t) -r(t)) + i(t) -c -p)V_x(t,x)}_{\Delta u(t,x)\mu (t,x)} \\ + \underbrace{\lambda(t)B(x+\frac{p}{\eta(t)},t) + U(t,x) - \lambda(t)V(t,x)}_{f(t,x,u(t,x), \sigma^T(t,x)\Delta u(t,x))}$$
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
# In the literature it was found that the solution of PDE and its spatial derivative are now the solution of the stochastic control problem {eq}`stoch_con`{}`weinan2020algorithms`. The relationship between the PDE {eq}`gen_form` and the BSDe {eq}`stoch_con` is based on the nonlinear Feynman-Kac formula {cite}`bloch2018machine` and {cite}`guler2019towards`. Under suitable additional regularity assumption on the nonlinearity $f$ in the sense that for all $t \in[0,T]$ it holds $\mathbb{P}$-a.s. that 
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
# 

# 