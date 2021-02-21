# https://blog.zakjost.com/post/adversarial_validation/
# https://alan-turing-institute.github.io/DataScienceTutorials.jl/end-to-end/boston-lgbm/

using CSV, DataFrames
using Printf, Random
using MLJ

LGBMRegressor = @load LGBMRegressor;

## Load data
df_train = outerjoin(
  CSV.read("./data/train_transaction.csv", DataFrame),
  CSV.read("./data/train_identity.csv", DataFrame),
  on=:TransactionID
);

df_test = outerjoin(
  CSV.read("./data/test_transaction.csv", DataFrame),
  CSV.read("./data/test_identity.csv", DataFrame),
  on=:TransactionID
);

test_names = names(df_test);
for (i, name) in enumerate(test_names)
  if (length(name)>=3) && (name[1:3]=="id-")
    test_names[i] = "id_"*name[4:end];
  end
end
rename!(df_test, test_names);

target = "dataset_label"
insertcols!(df_train, 1, target=>0);
insertcols!(df_test, 1, target=>1);

## Column specification
cat_cols = vcat(
  "ProductCD",
  ["card$i" for i in 1:6],
  "addr1", "addr2", "P_emaildomain", "R_emaildomain",
  ["M$i" for i in 1:9],
  "DeviceType", "DeviceInfo",
  ["id_$i" for i in 12:38]
);

numeric_cols = vcat(
  "TransactionAmt", "dist1", "dist2",
  ["C$i" for i in 1:14],
  ["D$i" for i in 1:15],
  ["V$i" for i in 1:339],
  [@sprintf "id_%02d" i for i in 1:11],
);

## Create adversarial data
function create_adversarial_data(df_train, df_test, cols; N_val=50000)
  df_master = vcat(df_train[:,cols], df_test[:,cols])

  samples_train = randperm(size(df_master,1))
  samples_val = samples_train[1:N_val]
  samples_train = samples_train[N_val+1:end]

  adversarial_val = df_master[samples_val,:]
  adversarial_train = df_master[samples_train,:]
  return adversarial_train, adversarial_val
end

features = vcat(cat_cols, numeric_cols, "TransactionDT");
all_cols = vcat(features, target);
adversarial_train, adversarial_test = create_adversarial_data(df_train, df_test, all_cols);

X_train = Matrix(adversarial_train[:, Not(:dataset_label)])
y_train = adversarial_train[:, :dataset_label]
X_test = Matrix(adversarial_test[:, Not(:dataset_label)])
y_test = adversarial_test[:, :dataset_label]

## LightGBM classifier
lgb = LGBMRegressor()
lgbm = machine(lgb, X_train, y_train)
boostrange = range(lgb, :num_iterations, lower=2, upper=500)
curve = learning_curve!(lgbm, resampling=CV(nfolds=5),
                        range=boostrange, resolution=100,
                        measure=rms)
