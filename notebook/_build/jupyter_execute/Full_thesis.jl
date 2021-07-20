using Plots

# the analytic result of the baseline model for CCRA utility functions to test if algorithm is working 

#parameters 
T=40
y(t)= 50000*exp(0.03*t)
r =0.04 
μ=0.09
σ= 0.18
ρ=0.03
λ(t)= 1/200 + 9/8000 * t
η(t)= 1/200 + 9/8000 * t 
γ = -3 






