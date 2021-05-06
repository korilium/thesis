# notebook Differentiation for hackers
# ref: https://github.com/MikeInnes/diff-zoo 
using MacroTools, Mjolnir, Flux, Zygote, DifferentialEquations
using SymbolicUtils, InteractiveUtils, SpecialFunctions


x= 2
y = x^2 +1 

y = :(x^2 +1)

typeof(y)

#tree structured 

y.head

y.args 

eval(y)

#obvious derivation 

function derive(ex, x)
    ex == x ? 1 :
    ex isa Union{Number, Symbol} ? 0 : 
    error("$ex is not differentiable")
end 

y = :(x)
derive(y, :x)

y = :(1)
derive(y, :x)

#checking which operation we can use with our derive function 
y = :(x +1)

@capture(y, a_ * b_)

@capture(y, a_ + b_)

a, b 

# add rule of additicity for derivatives 
function derive(ex, x)
    ex == x ? 1 : 
    ex isa Union{Number, Symbol} ? 0 : 
    @capture(ex, a_ + b_) ? :($(derive(a, x)) + $(derive(b, x))) : 
    erro("$ex is not differentiable")
end 

y = :(x +(1 + (x +1)))
derive(y, :x)

function derive(ex, x)
    ex == x ? 1 :
    ex isa Union{Number,Symbol} ? 0 :
    @capture(ex, a_ + b_) ? :($(derive(a, x)) + $(derive(b, x))) :
    @capture(ex, a_ * b_) ? :($a * $(derive(b, x)) + $b * $(derive(a, x))) :
    @capture(ex, a_^n_Number) ? :($(derive(a, x)) * ($n * $a^$(n-1))) :
    @capture(ex, a_ / b_) ? :($b * $(derive(a, x)) - $a * $(derive(b, x)) / $b^2) :
    error("$ex is not differentiable")
  end

  

y = :(3x^2 + (2x + 1))
dy = derive(y, :x)
#cleaning up 

addm(a, b) = a == 0 ? b : b == 0 ? a : :($a + $b)
mulm(a, b) = 0 in (a, b) ? 0 : a == 1 ? b : b == 1 ? a : :($a * $b)
mulm(a, b, c...) = mulm(mulm(a, b), c...)

function derive(ex, x)
    ex == x ? 1 :
    ex isa Union{Number,Symbol} ? 0 :
    @capture(ex, a_ + b_) ? addm(derive(a, x), derive(b, x)) :
    @capture(ex, a_ * b_) ? addm(mulm(a, derive(b, x)), mulm(b, derive(a, x))) :
    @capture(ex, a_^n_Number) ? mulm(derive(a, x),n,:($a^$(n-1))) :
    @capture(ex, a_ / b_) ? :($(mulm(b, derive(a, x))) - $(mulm(a, derive(b, x))) / $b^2) :
    error("$ex is not differentiable")
  end

  

derive(:(x / (1 + x^2)), :x)



#do not need to understand structure of printstruct

printstructure(x, _, _) = x

function printstructure(ex::Expr, cache = IdDict(), n = Ref(0))
  haskey(cache, ex) && return cache[ex]
  args = map(x -> printstructure(x, cache, n), ex.args)
  cache[ex] = sym = Symbol(:y, n[] += 1)
  println(:($sym = $(Expr(ex.head, args...))))
  return sym
end


:(1*2 + 1*2) |> printstructure;
:(x / (1 + x^2)) |> printstructure;


derive(:(x / (1 + x^2) * x), :x) |> printstructure;



include("utils.jl");
y = :(3x^2 + (2x + 1))

wy = wengert(y)