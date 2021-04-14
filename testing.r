library(diceR)
library(clue)
library(mlbench)


data("Cassini")

cassini_df = as.data.frame(Cassini)
cassini_df = subset (cassini_df, select = -classes)
cassini_df = data.matrix(cassini_df)

x <- dice(cassini_df, nk = 2:10, reps = 1, algorithms = "km", cons.funs ="kmodes", progress = FALSE)

pdf(file="cassini_dice.pdf")
plot(Cassini$x, col = x, xlab = "", ylab = "", main = "Consensus")
dev.off()

print(x)

"



partition_list = list()
list_i = 1
n_clusters = list(2,3,4,5,6,7,8,9,10)

#kmeans partition initialization
partition_list = list()
list_i = 1
while( list_i <= length(n_clusters) ){
    partition_list[[list_i]] <- kmeans(cassini_df, centers=initial_centers[1:n_clusters[[list_i]],], iter.max = 100)
    list_i = list_i + 1
}



#iteration over all different consensus methods
#method_list = list("SE","GV1", "DWH", "HE", "soft/manhattan", "hard/manhattan", "GV3", "soft/symdiff")
#consensus_output= list()

print(partition_list[1])

#
"