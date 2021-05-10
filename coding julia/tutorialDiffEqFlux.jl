
####### Introduction to Julia for Scientific Machine Learning #######
#ref: https://mitmath.github.io/18S096SciML/lecture2/ml 

#create project 
#]activate thesis
#add Flux DifferentialEquations DiffEqFlux

#reactivating project environment 
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate
using Flux, DifferentialEquations, DiffEqFlux, Statistics,
 Optim, DiffEqSensitivity, Plots, StochasticDiffEq, DiffEqBase.EnsembleAnalysis


###julia Linear Algebra Through Neural Networks 
W = [randn(32, 10), randn(32,32), randn(5,32)]
b = [zeros(32), zeros(32), zeros(5)]

NN(x) = W[3]*tanh.(W[2]*tanh.(W[1]*x + b[1]) + b[2]) +b[3]
NN(rand(10))

###training NN with flux 
NN2 = Chain(Dense(10, 32, tanh), 
            Dense(32,32, tanh), 
            Dense(32,5))
NN2(rand(10))

#replacing the scalar function 

NN3 = Chain(Dense(10, 32, x->x^2), 
            Dense(32,32, x->max(0,x)), 
            Dense(32,5))
NN3(rand(10))

###differential equations
function lotka!(du,u,p,t)
    x,y = u
    α,β,γ,δ = p
    du[1] = α*x - β*x*y
    du[2] = -δ*y + γ*x*y
end

u0 = [1.0,1.0]
p = [1.5,1.0,3.0,1.0]
tspan = (0.0,10.0)

using DifferentialEquations, Plots
prob = ODEProblem(lotka!,u0,tspan,p)
sol = solve(prob)
plot(sol)
plot(sol, vars=(1,2))

###optimizing Parameters of differential equations 
#save at each timestep 0.1 
sol = solve(prob, saveat=0.1)
data = Array(sol)

p =[1.6, 1.4, 2.0, 0.8]
_prob = remake(prob, p=p)
sol = solve(_prob, saveat=0.1)
plot(sol)
scatter!(sol.t, data')

#creating cost function 

function lotka_cost(p)
    _prob = remake(prob, u0 = convert.(eltype(p), u0), p=p)
    sol = solve(_prob, saveat=0.1, abstol=1e-10, reltol=1e-10, verbose =false)
    sol.retcode !== :Success && return Inf  #sometimes optimization does not convert 
    sqrt(sum(abs2, data - Array(sol)))
end 

lotka_cost(p)

lotka_cost([1.5,1.0,3.0,1.0])
using Optim 

res = Optim.optimize(lotka_cost, p, BFGS(), autodiff=:forward)

_prob = remake(prob, u0=convert.(eltype(p), u0), p=p)

sol = solve(_prob, saveat=0.1, abstol=1e-10, reltol=1e-10, verbose=false)

### solving ODEs with NN 

NNODE= Chain(x -> [x], 
            Dense(1,32, tanh), 
            Dense(32,1), 
            first)
NNODE(1.0)

#insert boundary condition of intial value

m(t) = t*NNODE(t) + 1f0

ϵ = sqrt(eps(Float32))
loss() = mean(abs2(((m(t+ϵ)-m(t))/ϵ) - cos(2π*t)) for t in 0:1f-2:1f0)


opt = Flux.Descent(0.01)
data = Iterators.repeated((), 5000)
iter = 0
cb = function () #callback function to observe training
  global iter += 1
  if iter % 500 == 0
    display(loss())
  end
end
display(loss())


Flux.train!(loss, Flux.params(NNODE), data, opt; cb=cb)

using Plots
t = 0:0.001:1.0
plot(t,g.(t),label="NN")
plot!(t,1.0 .+ sin.(2π.*t)/2π, label = "True")

######### Mixing Differential Equations and Machine Learning ########
#ref: https://mitmath.github.io/18S096SciML/lecture3/diffeq_ml


#create the ODE 
function lotka_volterra(du, u, p, t)
    x, y = u 
    α, β, δ, γ = p 
    du[1] = dx = α*x - β*x*y 
    du[2] = dy = -δ*y + γ*x*y
end 
u₀ = [1.0,1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lotka_volterra,u₀, tspan, p)
sol = solve(prob, Tsit5())
test_data = Array(solve(prob, Tsit5(), saveat = 0.1 ))
plot(sol)


#set up the NN 
p= [2.2, 1.0, 2.0, 0.4] #initial values Parameters 

function predict_adjoint()
    Array(concrete_solve(prob, Tsit5(), u0, p , saveat=0.1, abstol = 1e-6, reltol = 1e-5))
end 

loss_adjoint() = sum(abs2, predict_adjoint()-test_data)
iter = 0 
cb = function ()
    global iter += 1
    if iter % 60 ==0 
        display(loss_adjoint())

        pl = plot(solve(remake(prob, p=p), Tsit5(), saveat=0.0:0.1:10.0), lw=5, ylim=(0,8))
        display(scatter!(pl, 0.0:0.1:10.0, test_data', markersize=2))
    end 
end 


cb()

p = [2.2,1.0,2.0,0.4]

data = Iterators.repeated((), 300)
opt = ADAM(0.1)
Flux.train!(loss_adjoint, Flux.params(p), data, opt, cb=cb)

opt = Descent(0.00001)
Flux.train!(loss_adjoint, Flux.params(p), data, opt, cb=cb)


###defining and training Neural Ordinary Differential Equations 

# create fake data 
u0 = Flaot32[2.; 0.]
datasize= 30 
tspan = (0.0f0,1.5f0)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0;-2.0 -0.1]
    du .= ((u.^3)'true_A)'
end 

t = range(tspan[1], tspan[2], length = datasize)
prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob, Tsit5(), saveat=t))

#create NN 

dudt = Chain(x -> x.^3, 
    Dense(2, 50, tanh), 
    Dense(50,2))
#train 

p, re = Flux.destructure(dudt)
dudt2_(u, p, t) = re(p)(u)
prob = ODEProblem(dudt2_, u0, tspan, p)

function predict_n_ode()
    Array(concrete_solve(prob,Tsit5(),u0,p,saveat=t))
  end
  loss_n_ode() = sum(abs2,ode_data .- predict_n_ode())
  
  data = Iterators.repeated((), 300)
  opt = ADAM(0.1)
  iter = 0
  cb = function () #callback function to observe training
    global iter += 1
    if iter % 50 == 0
      display(loss_n_ode())
      # plot current prediction against data
      cur_pred = predict_n_ode()
      pl = scatter(t,ode_data[1,:],label="data")
      scatter!(pl,t,cur_pred[1,:],label="prediction")
      display(plot(pl))
    end
  end

cb()

ps = Flux.params(p)

Flux.train!(loss_n_ode, ps, data, opt, cb=cb)
########## Optimization of Ordinary Differential Equations ##########
#ref: https://diffeqflux.sciml.ai/dev/examples/optimization_ode/ 

function lotka_volterra(du, u, p, t)
    x, y = u 
    α, β, δ, γ = p 
    du[1] = dx = α*x - β*x*y
    du[2] = dy = -δ*y + γ*x*y 
end 

u₀ = [1.0,1.0]


tspan = (0.0,10.0)
tsteps = 0.0:0.1:10.0

p = [1.5, 1.0, 3.0, 1.0]

prob = ODEProblem(lotka_volterra, u₀, tspan, p)

sol = solve(prob)

plot(sol)

function loss(p)
    sol = solve(prob, Tsit5(), p=p, saveat = tsteps)
    loss = sum(abs2, sol.-1)
end 


callback = function (p, l, pred)
    display(l)
    plt = plot(pred, ylim = (0, 6))
    display(plt)
    # Tell sciml_train to not halt the optimization. If return true, then
    # optimization stops.
    return false
  end

  result_ode = DiffEqFlux.sciml_train(loss, p,
                                    ADAM(0.1),
                                    cb = callback,
                                    maxiters = 100)
                                    
remade_solution = solve(remake(prob, p = result_ode.minimizer), Tsit5(),      
                                    saveat = tsteps)
            plot(remade_solution, ylim = (0, 6))


###### Numerically Solving Partial Differential Equation ###### 

n = 9 
u0= randn(n) .+ 2.0

function heat_1d(du, u, p, t)
  dx2 = (1/(length(u)+1))^2
  for i in (2:length(u)-1)
    du[i] = (u[i-1] -2u[i] + u[i+1])/dx2 
  end 
  du[1] = (-2u[1] + u[2])/dx2
  du[end] = (u[end-1] - 2u[end])/dx2
end 

prob = ODEProblem(heat_1d, u0, (0.0,0.5))
sol = solve(prob, Tsit5(), saveat = 0.05)

plot(sol[1])
plot!(sol[1])
plot!(sol[3])
plot!(sol[4])
plot!(sol[10])


u₀ = randn(n) .+ 2.0
function heat_1d_d3(du, u, p, t)
  dx2 = (1/(length(u)+1))^2
  for i in 2:(length(u)-1)
    du[i]= (u[i-1]-2u[i] + u[i+1])/dx2
  end 
  du[1] = (3.0 + -2u[1] + u[2])/dx2
  du[end] = (u[end-1] - 2u[end])/dx2
end 

prob = ODEProblem(heat_1d_d3, u₀, (0.0,0.5))
sol = solve(prob, Tsit5(), saveat=0.05)

plot(sol[1])
plot!(sol[2])
plot!(sol[3])
plot!(sol[11])

#two dimensional heat equation 



n = 9
u0 = randn(n,n) .+ 2.0

function heat_2d(du,u,p,t)
  dx2 = (1/(size(u,1)+1))^2
  for i in 2:(size(u,1)-1), j in 2:(size(u,2)-1)
    du[i,j] = (u[i-1,j] + u[i,j-1] - 4u[i,j] + u[i+1,j] + u[i,j+1])/dx2
  end
  for i in 2:(size(u,1)-1)
    du[1,i] = (u[1,i-1] -4u[1,i] + u[2,i] + u[1,i+1])/dx2
    du[i,1] = (u[i-1,1] -4u[i,1] + u[i,2] + u[i+1,1])/dx2
    du[end,i] = (u[end-1,i] - 4u[end,i] + u[end,i+1] + u[end,i-1])/dx2
    du[i,end] = (u[i,end-1] - 4u[i,end] + u[i+1,end] + u[i-1,end])/dx2
  end

  du[1,1] = (u[2,1] + u[1,2] - 4u[1,1])/dx2
  du[1,end] = (u[2,end] + u[1,end-1] - 4u[1,end])/dx2
  du[end,1] = (u[end,2] + u[end-1,1] - 4u[end,1])/dx2
  du[end,end] = (u[end-1,1] + u[end,end-1] - 4u[end,end])/dx2
end

prob = ODEProblem(heat_2d, u0, (0.0,0.01))
sol = solve(prob,Tsit5(),saveat=0.001)

surface(sol[1])
surface(sol[3])
surface(sol[6])
surface(sol[9])
surface(sol[11])
##################### Stochastic Differential Equations, Deep Learning, and High-Dimensional PDEs###############

p = (μ = 1.0, σ = 1.0 )
f(u,p,t) = p.μ*u
g(u,p,t) = p.σ*u
prob = SDEProblem(f,g,1.0,(0.0,1.0), p)
sol = solve(prob, SRIW1())
plot(sol)

enprob = EnsembleProblem(prob)
# Note: Automatically parallel! Try on the GPU with EnsembleGPUArray() from DiffEqGPU.jl!
ensol = solve(enprob,SRIW1(),trajectories=10000)
summ = EnsembleSummary(ensol)
plot(summ)




### SDEs as Regularized ODEs and Neural Stochastic Differential Equations

#create true SDE
u0 = Float32[2.;0.]
datasize = 50 
tspan = (0.0f0, 1.0f0)

function trueSDEfunc(du,u, p, t)
  true_A = [-0.1 2.0; -2.0 -0.1]
  du .= ((u.^3)'true_A)'
end
t=range(tspan[1], tspan[2], length=datasize)
mp = Float32[0.2,0.2]
function true_noise_func(du,u,p,t)
  du .= mp.*u
end 

prob = SDEProblem(trueSDEfunc, true_noise_func, u0, tspan)

ensemble_prob = EnsembleProblem(prob)
ensemble_sol = solve(ensemble_prob, SOSRI(), trajectories = 10000)
ensemble_sum = EnsembleSummary(ensemble_sol)
sde_data, sde_data_vars = Array.(timeseries_point_meanvar(ensemble_sol,t))


#neural SDE 

drift_dudt = Chain(x -> x.^3, 
              Dense(2,50, tanh), 
              Dense(50,2))
diffusion_dudt = Chain(Dense(2,2))
n_sde = NeuralDSDE(drift_dudt, diffusion_dudt, 
                  tspan, SOSRI(), saveat =t , reltol =1e-1, abstol = 1e-1)
ps = Flux.params(n_sde)

pred = n_sde(u0) # Get the prediction using the correct initial condition
p1,re1 = Flux.destructure(drift_dudt)
p2,re2 = Flux.destructure(diffusion_dudt)
drift_(u,p,t) = re1(n_sde.p[1:n_sde.len])(u)
diffusion_(u,p,t) = re2(n_sde.p[(n_sde.len+1):end])(u)
nprob = SDEProblem(drift_,diffusion_,u0,(0.0f0,1.2f0),nothing)

ensemble_nprob = EnsembleProblem(nprob)
ensemble_nsol = solve(ensemble_nprob,SOSRI(),trajectories = 100, saveat = t)
ensemble_nsum = EnsembleSummary(ensemble_nsol)
p1 = plot(ensemble_nsum, title = "Neural SDE: Before Training")
scatter!(p1,t,sde_data',lw=3)
scatter(t,sde_data[1,:],label="data")
scatter!(t,pred[1,:],label="prediction")