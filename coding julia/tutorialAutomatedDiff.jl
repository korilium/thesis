# Forward-Mode Automatic Differentiation (AD) via High Dimensional Algebras
# ref: https://mitmath.github.io/18337/lecture8/automatic_differentiation
using InteractiveUtils, Base, StaticArrays, ForwardDiff
import Base
eps(Float64)

@show eps(1.0)
@show eps(0.01)

ϵ = 1e-10rand()
@show ϵ
@show (1+ϵ)
ϵ2 = (1+ϵ) - 1
(ϵ - ϵ2)

#creating dual number 

struct Dual{T}
  val::T
  der::T
end 

#import the operation Base 
import Base: + 
f::Dual + g::Dual = Dual(f.val - g.val, f.der -g.der)
#same as 
Base.:+(f::Dual, g::Dual) = Dual(f.val + g.val, f.der +g.der)
Base.:+(f::Dual, α::Number) = Dual(f.val + α, f.der)
Base.:+(α::Number, f::Dual) = f + α

#product rule 
Base.:*(f::Dual, g::Dual) = Dual(f.val*g.val, f.der*g.val + f.val*g.der)
Base.:*(α::Number, f::Dual) = Dual(f.val * α, f.der * α)
Base.:*(f::Dual, α::Number) = α * f

#quatient rule 
Base.:/(f::Dual, g::Dual) = Dual(f.val/g.val, (f.der*g.val -f.val*g.der)/(g.val^2))
Base.:/(f::Dual, α::Number) = Dual(f.val/α, f.der*(1/α))
Base.:/(α::Number,f::Dual ) = Dual(α/f.val, -α*f.der/f.val^2)

#higher orders 
Base.:^(f::Dual, n::Integer) = Base.power_by_squaring(f, n) 

#test our new dual number 

f = Dual(3, 4)
g = Dual(5, 6)
f+g
f*g

f*(g+g)

#skipping performance 
#defining Higher Order Primitives

import Base: exp 
exp(f::Dual) = Dual(exp(f.val), exp(f.val)*f.der)
f

exp(f)

#differentiating arbitrary functions 
h(x) = x^2 +2 
a = 3 

xx =Dual(a, 1)

h(xx)
derivative(f, x) = f(Dual(x, one(x))).der

derivative(x -> 3*x^5 +2, 2)


#higher dimensions 
ff(x, y) =  x^2 +x*y
a, b = 3.0, 4.0 

ff_1(x) = ff(x, b) #single variable function 

#defferentiate single variable function 

derivative(ff_1, a)

#under the hood 

ff(Dual(a, one(a)), b)

#defferentiaite with respect to y 

ff_2(y) = ff(a, y)

derivative(ff_2, b)


ff(a, Dual(b, one(b)))
#need to do two separate calculations for all the partial derivatives


###### implementation of higher-dimensional forward-mode AD 

#creating dual numbers in vector space 
struct MultiDual{N, T}
  val::T
  derivs::SVector{N,T}
end 

import Base: +, *


function +(f::MultiDual{N,T}, g::MultiDual{N,T}) where {N, T}
  return MultiDual{N,T}(f.val + g.val, f.derivs + g.derivs)
end

function *(f::MultiDual{N,T}, g::MultiDual{N,T}) where {N, T}
  return MultiDual{N,T}(f.val *g.val, f.val .* g.derivs +g.val .* f.derivs)
end 

gg(x, y) = x*x*y + x +y 

(a, b) = (1.0, 2.0)

xx = MultiDual(a, SVector(1.0, 0.0))
yy = MultiDual(b, SVector(0.0,1.0))

gg(xx, yy)

#Jacobian 

ff(x, y) = SVector(x*x + y*y, x+y)

ff(xx, yy)

# matrix implementation of  Forward mode AD in Forwarddiff package 

ForwardDiff.gradient(xx -> ((x, y) = xx; x^2 * y + x*y), [1,2])

#application solving nonlinear equations 



function Newton_step(f, x0)
  J= ForwardDiff.jacobian(f,x0)
  δ = J\ f(x0)   #julia uses backslash \ to solve linear systems 
  return x0 -δ
end 

function newton(f, x0)
  x = x0 

  for i in 1:10 
    x = Newton_step(f, x)
    @show x
  end 

  return x 
end 

ff(xx) = (( x, y) = xx; SVector(x^2 + y^2 -1, x-y))

x0 = SVector(3.0, 5.0)
x = newton(ff, x0)