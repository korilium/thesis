cd(@__DIR__)
using Pkg; Pkg.activate("DiffEq"); Pkg.instantiate()

using DifferentialEquations,
DifferentialEquations.EnsembleAnalysis,
ParameterizedFunctions,
BenchmarkTools,
Catalyst, 
StaticArrays, 
ParameterizedFunctions, 
Latexify




### first do the tutorials 
f(u, p,t) = 0.98u

u0 = 1.0 
tspan= (0.0, 1.0)
prob = ODEProblem(f, u0, tspan)

sol = solve(prob)

gr(); plot(sol)
plot!(sol.t, t -> 1.0*exp(0.98*t), ls = :dash, label="true solution ! ")

sol.t
sol.u

# the solutions in time points

[t+u for (u, t) in tuples(sol)]

sol = solve(prob, abstol=1e-8, reltol=1e-8)

plot(sol)
plot!(sol.t, t->1.0*exp(0.98t),lw=3,ls=:dash,label="True Solution!")


#choosing Solver Algorithms 
#stiff
sol = solve(prob, alg_hints=[:stiff])

#system of ODEs 

function lorenz!(du, u, p, t)
    σ, ρ, β = p 
    du[1] = σ*(u[2] - u[1])
    du[2] = u[1]*(ρ - u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3] 
end

u0 = [1.0,0.0,0.0]
p = (10, 28, 8/3)
tspan = (0.0, 100.0)

prob = ODEProblem(lorenz!, u0, tspan, p)

sol= solve(prob)

plot(sol)

plot(sol, vars=(1,2,3))

# internal Types 

A = [   1. 0 0 -5 
        4 -2 4 -3
        -4 0 0 1
        5 -2 2 3]
u0 = rand(4, 2)
tspan = (0.0, 1.0) 
f(u, p, t) = A*u 

prob = ODEProblem(f, u0, tspan)

sol = solve(prob)
sol[3]

plot(sol)

#### second tutorial: Choosing an ODE Algorithm ####



van! = @ode_def VanDerPol begin 
    dy = μ*((1-x^2)*y - x)
    dx = 1*y 
end μ
prob = ODEProblem(van!, [0.0,2.0], (0.0,6.3), 1e6)

sol = solve(prob, Tsit5()) # does not converge too stiff 

sol = solve(prob, alhg_hints = [:stiff])
plot(sol, denseplot = false)
plot(sol, ylims = (-10.0, 10.0))


function lorenz!(du,u,p,t)
    σ,ρ,β = p
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
end
u0 = [1.0,0.0,0.0]
p = (10,28,8/3)
tspan = (0.0,100.0)
prob = ODEProblem(lorenz!,u0,tspan,p)


@btime solve(prob)
@btime solve(prob, alg_hints = [:stiff])

function lorenz(u, p, t)
    dx = 10.0*(u[2]-u[1])
    dy = u[1]*(28.0 - u[3])- u[2]
    dz = u[1]*u[2] - (8/3)*u[3]
    [dx, dy, dz]
end 

u0 = [1.0; 0.0 ; 0.0]
tspan = (0.0, 100.0)
prob = ODEProblem(lorenz, u0, tspan)

@benchmark solve(prob, Tsit5())
@benchmark solve(prob, Tsit5(), save_everystep =false)
## needs to allocate memory for the vector create code such thta this does not happen
#see lorenz! to efficiently allocate memory
function lorenz!(du,u,p,t)
    du[1] = 10.0*(u[2]-u[1])
    du[2] = u[1]*(28.0-u[3]) - u[2]
    du[3] = u[1]*u[2] - (8/3)*u[3]
   end


prob = ODEProblem(lorenz!, u0, tspan)
@benchmark solve(prob, Tsit5())


A = @SVector [2.0, 3.0,5.0]

#using Static Arrays to optimize 

#### third tutorial callbacks and events


ball! = @ode_def BallBounce begin
  dy =  v
  dv = -g
end g

# create a condition 

function condition(u,t,integrator)
    u[1]
  end

#making the baouncing affect of the ball 

function affect!(integrator)
    integrator.u[2] = -integrator.p[2] * integrator.u[2]
end

bounce_cb = ContinuousCallback(condition, affect!)

u0 = [50.0, 0.0]
tspan = (0.0, 15.0)
p = (9.8, 0.9)
prob = ODEProblem(ball!, u0, tspan, p, callback = bounce_cb)

sol = solve(prob, Tsit5())

gr(); plot(sol)

#child kicking the ball 

function condition_kick(u, t, integrator)
    t == 2 
end 

function affect_kick!(integrator)
    integrator.u[2] += 50 
end 

kick_cb = DiscreteCallback(condition_kick, affect_kick!)
u0 = [50.0, 0.0]
tspan = (0.0, 10.0)
p= (9.8, 0.9)
prob = ODEProblem(ball!, u0, tspan,p, callback=kick_cb)

sol = solve(prob, Tsit5(), tstops=[2.0])
plot(sol)

cb = CallbackSet(bounce_cb,kick_cb)

u0 = [50.0, 0.0]
tspan= (0.0,15.0)
p = (9.8,0.9)
prob= ODEProblem(ball!, u0, tspan, p, callback=cb)
sol = solve(prob, Tsit5(), tstops= [2.0])
plot(sol)

### integration Termination and Directional handling

u0 = [1.,0.]
harmonic! = @ode_def HarmonicOscillator begin 
    dv = -x
    dx = v 
end 

tspan = (0.0, 10.0)
prob = ODEProblem(harmonic!, u0, tspan)
sol = solve(prob)
plot(sol)

function terminate_affect!(integrator)
    terminate!(integrator)
end 

function terminate_condition(u, t, integrator)
    u[2]
end 
terminate_cb = ContinuousCallback(terminate_condition, terminate_affect!)

sol = solve(prob, callback= terminate_cb)
plot(sol)


terminate_upcrossing_cb = ContinuousCallback(terminate_condition,terminate_affect!, nothing)
sol= solve(prob, callback = terminate_upcrossing_cb)

plot(sol)

#manifold Project 

tspan = (0.0, 10000.0)
prob = ODEProblem(harmonic!, u0, tspan)
sol = solve(prob)
gr(fmt=:png)
plot(sol, vars=(0,1), denseplot=false)




#### stochastic Differential Equations 

#scalar SDEs 
α = 1 
β = 1

u₀= 1/2 
f(u, p, t) = α*u 
g(u, p, t) = β*u 
dt = 1//2^4
tspan = (0.0, 1.0)
prob = SDEProblem(f,g, u₀, tspan)

sol = solve(prob, EM(), dt= dt)
plot(sol)

# using higher orders methods 

f_analytic(u₀, p, t, W) = u₀*exp((α - (β^2)/2)*t + β*W)
ff = SDEFunction(f, g, analytic = f_analytic)
prob = SDEProblem(ff, g, u₀, tspan)
sol = solve(prob, EM(), dt=dt)
plot(sol, plot_analytic=true)
sol = solve(prob, SRIW1(), dt=dt, adaptive=false)
plot(sol, plot_analytic=true)

#multiple simulations 
ensembleprob = EnsembleProblem(prob)

sol = solve(ensembleprob, EnsembleThreads(), trajectories = 1000)


summ= EnsembleSummary(sol, 0:0.01:1)
plot(summ, labels="Middle 95%")
summ= EnsembleSummary(sol, 0:0.01:1, quantiles=[0.25,0.75])
plot!(summ, labels="Middle 50%", legend=true)

#### Discrete Stochastic Equations 



sir_model = @reaction_network begin
    c1, s + i --> 2i
    c2, i --> r
end c1 c2


p = (0.1/1000, 0.01)
prob = DiscreteProblem(sir_model, [999,1,0], (0.0, 250.0), p)
jump_prob = jumpProblem(sir_model, prob, Direct())




############ PROBLEM 1: Investigating Sources of ndomness and Uncertainty in a Stiif Biological System #####

#part one 
function Oregonator( du,u, p, t)
    s, w, q = p
    du[1] = s*(u[2] - u[1]*u[2] + u[1] - q*u[1]^2)
    du[2] = (-u[2]- u[1]*u[2]+ u[3])/s
    du[3] = w*(u[1] - u[3])
end  
p = [77.27, 0.161, 8.375e-6]
u0 = [1.0;2.0;3.0]
tspan = (0.0, 500.0)

prob= ODEProblem(Oregonator, u0, tspan, p)




@btime solve(prob, Tsit5())
@btime solve(prob, Rodas5())
# Problem 1, Part two : adding stochasticity 

function multplication(du, u, p, t)
    s, w, q = p 
    du[1] = s*(u[2]- u[1]*u[2]+ u[1] - q*u[1^2])
    du[2] = (-u[2]- u[1]*u[2]+ u[3])/s
    du[3] = w*(u[1]- u[3])
end 

function sigma(du, u, p, t)
    du[1] = 0.1
    du[2] = 0.1 
    du[3] = 0.1 
end 

u₀ = [1 , 2, 3]

stoch_prob = SDEProblem(multplication, sigma, u₀, (0.0,30.0), p)
sol = solve(stoch_prob,SOSRI())

ensembleprob = EnsembleProblem(stoch_prob)
sol = solve(ensembleprob, EnsembleThreads(), trajectories=1000)
summ = EnsembleSummary(sol, 0:0.01:1)
plot(summ, labels = "Middle 95%")

### third part : Gillespie jump models of discrete stochasticity

A+Y -> X+P
X+Y -> 2P
A+X -> 2X + 2Z
2X  -> A + P (note: this has rate kX^2!)
B + Z -> Y