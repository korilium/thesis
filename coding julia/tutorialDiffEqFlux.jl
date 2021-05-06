using DifferentialEquations, DiffEqFlux, Flux, Optim, DiffEqFlux, DiffEqSensitivity, Plots


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