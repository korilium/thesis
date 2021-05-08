

####### Introduction to Julia for Scientific Machine Learning #######
#ref: https://mitmath.github.io/18S096SciML/lecture2/ml 





using Flux, Plots

######## basics ####### 
#ref: https://fluxml.ai/Flux.jl/stable/models/basics/ 

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

#simple models 
W = rand(2, 5)
b = rand(2)

predict(x)= W*x .+ b

function loss(x, y)
    ŷ=predict(x)
    sum(( y .-ŷ).^2)
end 

x, y = rand(5), rand(2)

loss(x, y)

#use gradient decent 

gs = gradient(() -> loss(x, y), params(W,b))

W⁻ = gs[W]

W.-= 0.1 .* W⁻ 

loss(x, y) 


#building Layers 

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

#handwritten 
Wxh = randn(5,2)
Whh = randn(5,5)
b = randn(5)

function rnn(h, x)
    h = tanh.(Wxh * x .+ Whh * h .+ b)
    return h, h
end 

x = rand(2)
h = rand(5)

h, y = rnn(h, x)

#with flux 

rnn = Flux.RNNCell(2,5)
x = rand(Float32, 2)
h =rand(Float32, 5)
h, y = rnn(h, x)

#stateful Models Recur wrapper 

x = rand(Float32, 2)
h = rand(Float32, 5)

m = Flux.Recur(rnn, h)

y = m(x)


RNN(2,5) # RNN is just a wrapper around Recur 

#creating a sequence so memory becomes usefull 
m = Chain(RNN(2, 5), Dense(5, 1), x -> reshape(x, :))

x= rand(Float32, 2)
m(x)

#muliple steps in one 
x=[rand(Float32, 2) for i = 1:3]
m.(x)





