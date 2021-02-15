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
