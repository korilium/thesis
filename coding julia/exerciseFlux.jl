
cd(@__DIR__)
using Pkg; Pkg.activate("Exercise"); Pkg.instantiate()
using Flux, Plots


W = [randn(32,10), randn(32,32), randn(5,32)]
b = [zeros(32), zeros(32), zeros(5)]

NN(x) = W[3]*tanh.(W[2]*tanh.(W[1]*x + b[1]) + B[2])+ b[3]

### in package documentation of Flux

f(x) = 3x^2 + 2x +1
df(x) = gradient(f, x)[1]

df(2)

df2(x) = gradient(df, x)[1]

df(2)

f(x, y) = sum((x .- y).^2)

gradient(f, [2, 1], [2, 0])

x = [4,2]; y = [2, 0]

gs = gradient(params(x, y)) do 
    f(x, y)
end 

gs[x]

gs[y]


f(x, y) = sum(x.^2 .+2 .* x .* y .+ y.^2) # need to have sum do not know why 

x = [2.0];
 y = [3]

gt = gradient(params(x, y)) do 
    f(x, y)
end

gt[x]

#simple Models 

W = rand(2, 5)
b = rand(2)

predict(x) = W*x .+ b


function loss(x, y)
    ŷ = predict(x)
    sum((y .- ŷ).^2)
end 

x, y = rand(5), rand(2)

loss(x, y)

#improve loss use gradient decent 

gs = gradient(() -> loss(x, y), params(W,b))

#train model 

Ŵ = gs[W]

W .-= 0.1 .* Ŵ

loss(x, y) ## loss increases at a certain time point why?? 

#complex model 

W1 = rand(3, 5)
b1 = rand(3)
layer1(x) = W1 * x .+ b1


W2 = rand(2, 3)
b2 = rand(2)
layer2(x) = W2 * x .+ b2

model(x) = layer2(σ.(layer1(x)))

model(rand(5))

# function that creates linear layers 

function linear(in , out)
    W = randn(out, in)
    b = randn(out)
    x -> W * x .+ b
end 

linear1 = linear(5, 3)
linear2 = linear(3, 2)

model(x) = linear2(σ.(linear1(x)))


model(rand(5))


# with affine layers which Flux uses 

struct affine
    W
    b
  end
  
  affine(in::Integer, out::Integer) =
    affine(randn(out, in), randn(out))
  
  # Overload call, so the object can be used as a function
  (m::affine)(x) = m.W * x .+ m.b
  
  a = affine(10, 5)
  
  a(rand(10)) # => 5-element vector


  #stacking it up 

  layers = [Dense(10, 5, σ), Dense(5, 2), softmax]

  models(x) = foldl((x, m) -> m(x), layers, init =x )

  model(rand(10))


  model2 = Chain(
      Dense(10, 5, σ), 
      Dense(5, 2), 
      softmax
  )

  model2(rand(10))

  #### recurrent Models ####

Wxh = randn(5, 10)
Whh = randn(5, 5)
b = randn(5)

function rnn(h, x)
    h = tanh.(Wxh * x .+ Whh * h .+ b)
    return h, h
end 
x = rand(10)
h = rand(5)
h, y = rnn(h, x)


# package function 

rnn2 = Flux.RNNCell(10, 5)

x= rand(10)
h = rand(5)

h, y = rnn2(h, x)

#stateful models (don't manage hidden states)

x = rand(10)
h = rand(5)

m = Flux.Recur(rnn, h)

y= m(x)

RNN(10,5)
# Sequences 

seq = [rand(10) for i = 1:10 ]

m.(seq)

