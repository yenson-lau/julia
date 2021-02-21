using CSV, DataFrames

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

insertcols!(df_train, 1, "dataset_label"=>0);
insertcols!(df_test, 1, "dataset_label"=>1);

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