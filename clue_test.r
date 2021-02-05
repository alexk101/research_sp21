library(clue)

data("Cassini")
## Plot the data set:
plot(Cassini$x, col = as.integer(Cassini$classes),
xlab = "", ylab = "")
## Create a "random" k-means partition of the data:
set.seed(1234)

clusterings <- list()

for(i in 2:9) 
{
    clusterings=append(clusterings, as.cl_partition(kmeans(Cassini$x, i)))
}

ensemble = cl_ensemble(list = clusterings)
consensus <- cl_consensus(ensemble, method = "SE" ,weights = 1, control = list()) 
## And plot that.
plot(Cassini$x, col = cl_class_ids(consensus),
xlab = "", ylab = "")
## (We can see the problem ...)
