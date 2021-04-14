library(diceR)
library(clue)
library(mlbench)
library(stats)

data("Cassini")

cassini_df = as.data.frame(Cassini)
cassini_df = subset (cassini_df, select = -classes)
cassini_df = data.matrix(cassini_df)

#####################Library Functions####################

"%||%" <- function(a, b) {
  if (!is.null(a)) a else b
}

init_array <- function(data, r, a, k) {
  rn <- rownames(data) %||% seq_len(nrow(data))
  dn <- list(rn, paste0("R", seq_len(r)), a, k)
  array(NA_integer_, dim = purrr::map_int(dn, length), dimnames = dn)
}

cc_other <- function(data, nk, reps, algs, n) {
  alg <- toupper(algs)
  arr <- init_array(data, reps, alg, nk)

  initial_centers = read.csv("initial_centers.csv", header=FALSE)
  initial_centers = data.matrix(initial_centers)
  list_i = 1
  n_clusters = list(2,3,4,5,6,7,8,9,10)

  for (j in seq_along(algs)) {
    for (k in seq_along(nk)) {
      for (i in seq_len(reps)) {
        ind.new <- sample(n, n)
        x <- data[ind.new, ]
        arr[ind.new, i, j, k] <- as.integer(kmeans(data, centers=initial_centers[1:n_clusters[[list_i]],], iter.max = 100)$cluster)
        list_i = list_i + 1
      }
    }
  }
  arr
}

consensus_trash = cc_other(data=cassini_df, nk=2:10, reps=1, algs=c("km"), n=nrow(cassini_df))

cons=list()

cons[[1]]= CSPA(consensus_trash, 3)
#cons[[1]] = k_modes(E=consensus_trash[, , 1, 1, drop=FALSE])
#cons[[3]] = LCA(E=consensus_trash[, , 1, 1, drop=FALSE])
#cons[[4]] = LCE(E=consensus_trash[, , 1, 1, drop=FALSE], k=3, sim.mat="cts")
#cons[[5]] = majority_voting(E=consensus_trash[, , 1, 1, drop=FALSE])

print(cons[[1]])

pdf(file="testing.pdf")
for(x in cons) {
    plot(Cassini$x, col = x, xlab = "", ylab = "", main = "Consensus 1")
}
dev.off()

#length(consensus)
#print(consensus)

############################################################
