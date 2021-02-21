import RDatasets
import MLJ

iris = RDatasets.dataset("datasets", "iris");
y, X = MLJ.unpack(iris, ==(:Species), colname->true);

# first(X, 3) |> MLJ.pretty
# MLJ.models(MLJ.matching(X,y))

## Tree model
Tree = @MLJ.load DecisionTreeClassifier;
tree = Tree();
mach = MLJ.machine(tree, X, y)

## Train
train, test = MLJ.partition(eachindex(y), 0.7, shuffle=true);
MLJ.fit!(mach, rows=train);

## Evaluate
MLJ.evaluate!(mach,
  resampling=MLJ.Holdout(fraction_train=0.7, shuffle=true),
  measures=[MLJ.log_loss, MLJ.brier_score],
  verbosity=0
)