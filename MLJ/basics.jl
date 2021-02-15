## Split-Apply-Combine
using RDatasets, DataFrames
using Statistics

# groupby
iris = dataset("datasets", "iris");
gdf = groupby(iris, :Species);
subdf_setosa = gdf[1];
describe(subdf_setosa, :min, :mean, :max)

# combine
df = DataFrame(a=1:3, b=4:6);
foo(v) = v[1:2] .* 2;
combine(df, :a=>sum, :b=>foo, nrow)

# combine + groupby
# 1) group iris df by Species
# 2) apply mean / std to :PetalLength for each subdf
# 3) rename as MPL / SPL
combine(groupby(iris, :Species), :PetalLength => mean => :MPL,
                                 :PetalLength => std => :SPL)

# an also broadcast multiple columns to one function
combine(gdf, names(iris, Not(:Species)) .=> std)



## Categorical vectors
using CategoricalArrays

# a column of categorical feature values
v = categorical(["AA", "BB", "CC", "AA", "BB", "CC"]);
v_categories = levels(v)  # returns unique values in order

# categoricals can also be ordered, e.g. default lexicographic order
v = categorical([1, 2, 3, 1, 2, 3, 1, 2, 3], ordered=true);

# or specify order manually
v = categorical(["high", "med", "low", "high", "med", "low"], ordered=true);
new_order = ["low", "med", "high"];
levels!(v, new_order);

# levels of categorical vectors disregard missing values
v = categorical(["AA", "BB", missing, "AA", "BB", "CC"]);
levels(v)



## Scientific types: Continuous, Count(able), OrderedFactor
using MLJScientificTypes

boston = dataset("MASS", "Boston");
sch = schema(boston);

# Note unique(boston.Chas) is just [0,1]... should be OrderedFactor
boston2 = coerce(boston, :Chas=>OrderedFactor, :Rad=>OrderedFactor);
println(eltype(boston2.Chas))     # CategoricalArrays.CategoricalValue{Int64,UInt32}
println(elscitype(boston2.Chas))  # OrderedFactor{2}
println();

# Strings default to Textual scitype, can coerce or drop
feature = ["AA", "BB", "AA", "AA", "BB"]
println(elscitype(feature));

feature2 = coerce(feature, Multiclass)
println(elscitype(feature2));
println();

# We can also coerce entire datatypes
data = select(boston, [:Rad, :Tax])
data2 = coerce(data, Count => Continuous)
schema(data2)

# We can also coerce using autotypes
#   :few_to_finite, :discrete_to_continuous, :string_to_multiclass
boston3 = coerce(boston, autotype(boston, :few_to_finite))
schema(boston3)
