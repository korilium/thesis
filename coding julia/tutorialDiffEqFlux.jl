
####### Introduction to Julia for Scientific Machine Learning #######
#ref: https://mitmath.github.io/18S096SciML/lecture2/ml 

#create project 
#]activate thesis
#add Flux DifferentialEquations DiffEqFlux

#reactivating project environment 
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate
using Flux, DifferentialEquations, DiffEqFlux, Statistics,
 Optim, DiffEqSensitivity, Plots


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


uà = randn(n) .+ 2.0




