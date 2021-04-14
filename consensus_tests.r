x <- suppressWarnings
x(library(dplyr))
x(library(clue))
x(library(mlbench))
x(library(kernlab))
x(library(rlist))
x(library(quadprog))
x(library(lpSolve))
x(library(relations))

data("Cassini")

cassini_df = as.data.frame(Cassini)
cassini_df = subset (cassini_df, select = -classes)
cassini_df = data.matrix(cassini_df)

partition_list = list()
list_i = 1
n_clusters = list(2,3,4,5,6,7,8,9,10)

initial_centers = read.csv("initial_centers.csv", header=FALSE)
initial_centers = data.matrix(initial_centers)

#kmeans partition initialization
partition_list = list()
list_i = 1
while( list_i <= length(n_clusters) ){
    partition_list[[list_i]] <- kmeans(cassini_df, centers=initial_centers[1:n_clusters[[list_i]],], iter.max = 100)
    list_i = list_i + 1
}


consensus_output = cl_consensus(cl_ensemble(list = partition_list), method = "GV3", weights = 1, control = list())
print(consensus_output)
"
#iteration over all different consensus methods
method_list = list("SE","GV1", "DWH", "HE", "soft/manhattan", "hard/manhattan", "GV3", "soft/symdiff")
consensus_output= list()
count = 1
for(x in method_list) {
    consensus_output[[count]] = cl_consensus(cl_ensemble(list = partition_list), method = x, weights = 1, control = list())
    print(paste("Finished method ",count))
    count = count +1
    
}
count = 1

pdf(file="cassini-clue-consensus-scatter-plot.pdf")
for(i in consensus_output) {
    plot(Cassini$x, col = cl_class_ids(i), xlab = "", ylab = "", main = paste("Consensus with method: ",method_list[[count]]))
    count = count + 1
}
dev.off()
"