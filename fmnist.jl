using Flux
using Flux: onehotbatch, onecold, @epochs
using Flux: Data.DataLoader, Losses.logitcrossentropy
using MLDatasets: FashionMNIST
using Parameters: @with_kw

Xtrain, ytrain = FashionMNIST.traindata()
Xtest, ytest = FashionMNIST.testdata()
