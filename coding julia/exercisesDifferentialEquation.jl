using DifferentialEquations, DifferentialEquations.EnsembleAnalysis
using Distributions, DynamicHMC

#######Problem 1: Investigating Sources of Randomness and Uncertainty in a Stiff Biological System (B)####

#part 1: Simulating the Oregonator ODE model

function oregonator(du, u, p, t)
    s, w, q, = p 
    du[1] = s*(u[2]-u[1]*u[2] + u[1] - q*u[1]^2)
    du[2] = (-u[2]- u[1]*u[2] + u[3])/s
    du[3] = w*(u[1]-u[3])
end 

u₀ = [1.0, 2.0, 3.0]
p  = [77.27, 0.161, 8.375e-6]
timespan = (0.0, 360.0)

prob= ODEProblem(oregonator, u₀, timespan, p)

sol = solve(prob)

plot(sol)
plot(sol, vars = (1,2,3))

#part 2: investigating Stiffness 
timespan_stiff = [0.0, 500.0]
prob_stiffness = ODEProblem(oregonator!, u₀, timespan_stiff, p)
@btime sol_non_stiff = solve(prob_stiffness,Tsit5())
@btime sol_stiff = solve(prob_stiffness, Rodas5())

#no solutions for part 3 and 4 => do not bother making them 

#part 5: Adding Stochasticity with stochastic differential equations 



function oregonator(du,u,p,t)
    s,q,w = p
    y1,y2,y3 = u
    du[1] = s*(y2+y1*(1-q*y1-y2))
    du[2] = (y3-(1+y1)*y2)/s
    du[3] = w*(y1-y3)
end

u₀ = [1.0, 2.0, 3.0]
p  = [77.27, 0.161, 8.375e-6]
timespan = (0.0, 30.0)

function g(du, u, p, t)
    du[1]= 0.1u[1]
    du[2]= 0.1u[2]
    du[3]= 0.1u[3]
end 

prob_stoch = SDEProblem(oregonator, g, u₀, timespan, p)
sol_stoch  = solve(prob_stoch, SOSRI())


plot(sol_stoch)
plot(sol_stoch, vars = (1,2,3))


ensembleprob = EnsembleProblem(prob_stoch)

sol = solve(ensembleprob, EnsembleThreads(), trajectories=1000)
summ = EnsembleSummary(sol, 0:0.01:1)
plot(summ,label = "middel 95%")

# no solution part 6 => do not bother 

# part 7: Probabilistic Programming / Bayesian Parameter Estimation with DiffEqBayes.jl
t = 0.0:1.0:30.0
data = [1.0 2.05224 2.11422 2.1857 2.26827 2.3641 2.47618 2.60869 2.7677 2.96232 3.20711 3.52709 3.97005 4.64319 5.86202 9.29322 536.068 82388.9 57868.4 1.00399 1.00169 1.00117 1.00094 1.00082 1.00075 1.0007 1.00068 1.00066 1.00065 1.00065 1.00065
        2.0 1.9494 1.89645 1.84227 1.78727 1.73178 1.67601 1.62008 1.56402 1.50772 1.45094 1.39322 1.33366 1.2705 1.19958 1.10651 0.57194 0.180316 0.431409 251.774 591.754 857.464 1062.78 1219.05 1335.56 1419.88 1478.22 1515.63 1536.25 1543.45 1539.98
        3.0 2.82065 2.68703 2.58974 2.52405 2.48644 2.47449 2.48686 2.52337 2.58526 2.67563 2.80053 2.9713 3.21051 3.5712 4.23706 12.0266 14868.8 24987.8 23453.4 19202.2 15721.6 12872.0 10538.8 8628.66 7064.73 5784.29 4735.96 3877.66 3174.94 2599.6]
