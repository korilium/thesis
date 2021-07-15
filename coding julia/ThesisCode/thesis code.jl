cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate
using NeuralPDE, Flux, DifferentialEquations, LinearAlgebra, Plots, 
 DifferentialEquations.EnsembleAnalysis


#parameters 

T= 40 
y(t) = 50000*exp(0.03*t)
r(t)=exp(0.04f0*t)
μ(t)=0.09f0*t
σ=0.18f0
ρ(t)=0.03f0*t
λ(t) = 1/200 + 9/8000*t
η(t) = 1/200 + 9/8000*t
γ = -3 

# life insurance parameters 

t = 1:0.1:40

F(t) = exp(-0.0005625*t^2 - 0.005*t )
F(s,t) = F(s)/F(t)
f(t) = 1/200 + 9/8000*t * exp(-0.0005625*t^2 - 0.005*t)

plot(t,F)
plot(t,f)

### stochastic wealth equation 
#stock price evolution over 1 year 
u₀ = 1
f(u, p, t) = u 
g(σ, p, t) = σ
dt = 1//2^(4) 
tspan = (0.0, 1.0) 
prob = SDEProblem(f,g,u₀, tspan)
sol = solve(prob, EM(), dt=dt)
plot(sol)

ensembleprob = EnsembleProblem(prob)
sol = solve(ensembleprob,EnsembleThreads(),trajectories=1000)
summ = EnsembleSummary(sol,0:0.01:1)
plot(summ,labels="Middle 95%")
summ = EnsembleSummary(sol,0:0.01:1;quantiles=[0.25,0.75])
plot!(summ,labels="Middle 50%",legend=true)

#wealth evolution over 1 year
#set arbitrary parameters 
c =  1300   # consumption  
pr =  100   # premium 
i = 2000    # income 
θ = 10000   # dollar amount in risky assets 
u₀ = 20000  # intial wealth 

f(u, p, t) = r*u -c - pr + i + θ*(μ -r) 
g(u, p, t) = σ*θ
dt = 1//250
tspan = (0.0, 1.0) 
prob_Wealth = SDEProblem(f,g,u₀, tspan)
sol_Wealth = solve(prob_Wealth, EM(), dt=dt)
plot(sol_Wealth)

ensembleprob = EnsembleProblem(prob_Wealth)
sol_Wealth = solve(ensembleprob,EnsembleThreads(),trajectories=1000)
summ_Wealth = EnsembleSummary(sol_Wealth,0:0.01:1)
plot(summ_Wealth,labels="Middle 95%")
summ_Wealth = EnsembleSummary(sol_Wealth,0:0.01:1;quantiles=[0.25,0.75])
plot!(summ_Wealth,labels="Middle 50%",legend=true)


## how much c, p and theta is optimal??? 

#utility functions 

U(c,t) = exp(-ρ*t)/γ*c^γ
B(Z,t) = exp(-ρ*t)/γ*Z^γ
L(x) = exp(-ρ*T)/γ*x^γ 

# max expected utility 




d = 1 # number of dimensions
X0 = fill(0.0f0,d )
tspan = (0.0f0,1.0f0)

g(x) = (exp(-ρ*T)/γ)*x^γ
f(x, u, σᵀ∇u, p, t) = 0
μ_f(x,p,t) = (r + η)*x + i 
σ_f(x,p,t) = 0
prob = TerminalPDEProblem(g,f, μ_f, σ_f, X0, tspan, p = nothing)





