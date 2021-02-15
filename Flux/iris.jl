using Random
using Flux
using Flux: onehotbatch, onecold, @epochs, Data.DataLoader, Losses.logitcrossentropy
using Parameters: @with_kw
using PyCall

Random.seed!(1234)
iris = pyimport("sklearn.datasets")["load_iris"]()

@with_kw mutable struct Args
  η::Float64 = 1e-1       # learning rate
  batchsize::Int = 10     # batch size
  epochs::Int = 10        # number of epochs
end
args = Args()

# Data should have size (n1, n2, ..., n_samples)
X = copy(iris["data"]')
y = iris["target"]
n_samples = size(X,2)

shuffle_order = shuffle(1:n_samples)
X = X[:,shuffle_order]
y = y[shuffle_order]

n_train = Int(n_samples*0.8)
Xtrain, ytrain = X[:,1:n_train], y[1:n_train]
Xtest, ytest = X[:,n_train+1:end], y[n_train+1:end]
Ytrain, Ytest = onehotbatch(ytrain, [0,1,2]), onehotbatch(ytest, [0,1,2])

# Batched data should have size (n1, n2, ..., batchsize)
train_data = DataLoader((Xtrain, Ytrain, ytrain), batchsize=args.batchsize)
test_data = DataLoader((Xtest, Ytest, ytest), batchsize=args.batchsize)

# Simple MLP
struct MLP
  sizes::Array{Integer,1}
  model::Chain

  MLP(sizes) = new(sizes, Chain(
    Dense(sizes[1], sizes[2], relu),
    Dense(sizes[2], sizes[end])
  ))
end
mlp = MLP([size(X,1), 5, length(unique(y))])

# Metrics
loss(X,Y,y) = logitcrossentropy(mlp.model(X), Y)

function loss_all(dataloader, model)
  out = 0.0
  for (X,Y,y) in dataloader
      out += logitcrossentropy(model(X), Y)
  end
  return out/length(dataloader)
end

function accuracy(data_loader, model)
  acc = 0.0
  for (X,Y,y) in data_loader
      acc += sum(onecold(model(X), [0,1,2]) .== y) / size(X,2)
  end
  return acc/length(data_loader)
end

# Training loop
opt = ADAM(args.η)
evalcb = () -> @show(loss_all(train_data, mlp.model))
function step()
  Flux.train!(loss, params(mlp.model), train_data, opt)
  evalcb()
end
@epochs args.epochs step()

println("")
@show accuracy(train_data, mlp.model)
@show accuracy(test_data, mlp.model)
;