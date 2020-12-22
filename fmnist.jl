using CUDA
using Flux
using Flux: onehotbatch, onecold, unsqueeze, @epochs
using Flux: Data.DataLoader, Losses.logitcrossentropy
using MLDatasets: FashionMNIST
using Parameters: @with_kw

if has_cuda()		          # Check if CUDA is available
  @info "CUDA is on"
  CUDA.allowscalar(false)
end

@with_kw mutable struct Args
  η::Float64 = 3e-2       # learning rate
  batchsize::Int = 128    # batch size
  epochs::Int = 20        # number of epochs
  device::Function = gpu  # set as gpu if available
end
args = Args()

function load_data(batchsize::Integer)
  Xtrain, ytrain = FashionMNIST.traindata()
  Xtest, ytest = FashionMNIST.testdata()

  Xtrain = Float32.(unsqueeze(Xtrain,3))
  Xtest = Float32.(unsqueeze(Xtest,3))

  ytrain.+=1;  ytest.+=1
  labels = sort(unique(ytest))


  train_data = DataLoader(
    (Xtrain, onehotbatch(ytrain, labels), ytrain),
    batchsize=batchsize, shuffle=true
  )

  test_data = DataLoader(
    (Xtest, onehotbatch(ytest, labels), ytest),
    batchsize=batchsize
  )

  return train_data, test_data
end

struct CNN
  imsize::Tuple{Integer,Integer,Integer}
  nclasses::Integer
  output_size::Tuple{Integer,Integer,Integer}
  model::Chain
end

# TODO: go over the size calculations for CNN
function CNN(data::DataLoader)
  imsize = (size(data.data[1])[1:2]..., 1)    # greyscale image
  nclasses = size(data.data[2],1)
  output_size = @. Int(floor((imsize[1]/8, imsize[2]/8, 32)))

  model = Chain(
    # First convolution, operating upon a 28x28 image
    Conv((3, 3), imsize[3]=>16, pad=(1,1), relu),
    MaxPool((2,2)),

    # Second convolution, operating upon a 14x14 image
    Conv((3, 3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    # Third convolution, operating upon a 7x7 image
    Conv((3, 3), 32=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    # Reshape 3d tensor into a 2d one using `Flux.flatten`, at this point it should be (3, 3, 32, N)
    flatten,
    Dense(prod(output_size), nclasses)
  )

  return CNN(imsize,nclasses,output_size,model)
end

function CNN(cnn::CNN, convert_model::Function)
  return CNN(cnn.imsize, cnn.nclasses, cnn.output_size, convert_model(cnn.model))
end

# Metrics
loss(X,Y,y) = logitcrossentropy(cnn.model(X), Y)

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
      acc += sum(onecold(model(X)) .== y) / size(X,2)
  end
  return acc/length(data_loader)
end

# TODO: Training loop, put on GPU
train_data, test_data = load_data(args.batchsize)
cnn = CNN(train_data)

opt = ADAM(args.η)
evalcb = () -> @show(loss_all(train_data, cnn.model))
function step()
  Flux.train!(loss, params(cnn.model), train_data, opt)
  evalcb()
end
@epochs args.epochs step()

println("")
@show accuracy(train_data, cnn.model)
@show accuracy(test_data, cnn.model)
;
