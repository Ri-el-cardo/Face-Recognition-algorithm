library(bmp)
library(OpenImageR)
library(ggplot2)
library(ggpubr)

names = list.files(pattern = "bmp")
data = matrix(0, length(names),165*120*3)
for (i in 1:length(names)){
  Im = read.bmp(names[i])
  red = as.vector(Im[,,1])
  green = as.vector(Im[,,2])
  blue = as.vector(Im[,,3])
  data[i,] = t(c(red, green, blue))
}

labels = matrix(0,480,1)
for (i in 1:480) {
  labels[i] = (i+5)%/%6
}
{
  #a function that returns a list with the trainData scaled and the testData scaled but respect to the
  #mean and variances of the varaibles with the observations of the train set 
  normalize_data = function(trainData, testData){
    train.mean = apply(trainData,2,mean)
    
    train.centered = trainData - t(matrix(c(train.mean),dim(trainData)[2],dim(trainData)[1]))
    test.centered = testData - t(matrix(c(train.mean),dim(trainData)[2],dim(testData)[1]))
    
    train.var = sqrt(apply(trainData, 2, var))
    train.scaled = matrix(0,dim(trainData)[1],dim(trainData)[2])
    for(i in 1:dim(train.scaled)[2]){
      train.scaled[,i] = train.centered[,i]/train.var[i]
    }
    
    test.scaled = matrix(0,dim(testData)[1],dim(testData)[2])
    for(i in 1:dim(test.scaled)[2]){
      test.scaled[,i] = test.centered[,i]/train.var[i]
    }
    
    return(list(train.scaled = train.scaled,
                test.scaled = test.scaled ))
  }  
  
  #a distance function that computes diferent types of distances given 2 vectors of the same length
  {
    distances = c("euclid","modified_SSE","canberra",
                  "simplified_mahalanobis","weighted_angle_based","angle_based",
                  "manhattan")
    distance = function(x,y,type_distance,eigenvalues = 0){
      if(type_distance == "euclid"){
        z = x-y
        total = sqrt(t(z)%*%z)
        return(as.numeric(total))
      }
      if(type_distance == "modified_SSE"){
        z = x-y
        numerator = t(z)%*%z
        denominator = (t(x)%*%x)*(t(y)%*%y)
        total = numerator/denominator
        return(as.numeric(total))
        
      }  
      if(type_distance == "canberra"){
        total = 0
        for (i in 1:length(x)){
          total = total + (abs(x[i]-y[i])/(abs(x[i])+abs(y[i])))
        }
        return(as.numeric(total))
      }
      if(type_distance == "simplified_mahalanobis"){
        total = 0
        
        for (i in 1:length(x)) {
          total = total + (x[i]*y[i]*sqrt(1/eigenvalues[i]))
          
        }
        total = - total
        return(as.numeric(total))
      }
      if(type_distance == "weighted_angle_based"){
        numerator = 0
        
        for(i in 1:length(x)) {
          numerator = numerator + (x[i]*y[i]*sqrt(1/eigenvalues[i]))
          
        }
        denominator = (t(x)%*%x)*(t(y)%*%y)
        total = numerator/sqrt(denominator)
        total = -total
        return(as.numeric(total))
      }
      if(type_distance == "angle_based"){
        numerator = 0
        
        for(i in 1:length(x)) {
          numerator = numerator + (x[i]*y[i])
          
        }
        denominator = (t(x)%*%x)*(t(y)%*%y)
        total = numerator/sqrt(denominator)
        total = -total
        return(as.numeric(total))
      }
      if(type_distance == "manhattan"){
        total = 0
        for(i in 1:length(x)) {
          total = total + abs(x[i]-y[i])
        }
        return(as.numeric(total))
      }
      
    }
    
    {
      list_distances = function(data, type_of_distance, objective, eigenvalues){
        n = dim(data)[1]
        index = matrix(rep(0),n)
        for (i in 1:n) {
          index[i] = distance(data[i,],objective ,type_of_distance, eigenvalues)
        }
        
        return(index)
      }
      
      knn = function(data, labels, type_of_distance = "euclid", objective, eigenvalues = 0, k = 1, threshold = 1000){
        
        vector.distances = list_distances(data, type_of_distance,objective, eigenvalues)
        if(min(vector.distances) >= threshold){
          return("0")
        }
        list.index = order(vector.distances, decreasing = F)[1:k]
        k.labels = labels[list.index]
        k.labels = table(k.labels)
        return(names(which.max(k.labels)))
      }
    }
}


#generates a dataframe in which the variables are the partitions of the dataset and the hyperparameters 
#in order to add later the accuracy of the model with every combination of this variables 
{
  partition = c(0,1,2,3,4,5)
  variances = c(0.9,0.91,0.92,0.93,0.94,0.95,0.96)
  ks = c(1,3,5)
  distances = c("euclid","modified_SSE","canberra",
                "simplified_mahalanobis","weighted_angle_based","angle_based",
                "manhattan")
}
}

#this generates indexes in which every 6 indexes(each group corresponds to the same person in the trainset) are randomly rearranged
rearranged.indexes = as.vector(replicate(80,sample(6))) + sapply(0:479, function(i)floor(i/6)*6) 

#rearranged.indexes[seq(1,480,6)+i]
#when i ranges from 0:5, those five sets are a partition of the dataset in which there is 1 observation of each person



#the accuracy vector will be added to the accurate dataframe once it is filled with the 
#prediction accuracy of each combination of the hyperparameters
accuracy = vector()

for(i in partition){
  test.indexes = rearranged.indexes[seq(1,480,6)+i]
  
  train = data[-test.indexes,]
  test = data[test.indexes,]
  train.labels = labels[-test.indexes]
  test.labels = labels[test.indexes]
  
  data.separated = normalize_data(train,test)
  
  train.scaled = data.separated$train.scaled
  test.scaled = data.separated$test.scaled
  
  # Sigma = cov(train.scaled)
  Sigma_ = train.scaled%*%t(train.scaled)/(nrow(train.scaled)-1)
  
  Eigen = eigen(Sigma_)
  Eigenvalues = Eigen$values
  Prop.Var = Eigenvalues/sum(Eigenvalues)
  
  Cummulative.Var = cumsum(Eigenvalues)/sum(Eigenvalues)
  
  for(j in variances){
    
    
    n_eigen = min(which(Cummulative.Var>j))
    eigenvalues = Eigen$values[1:n_eigen]
    
    Eigenvectors = Eigen$vectors[,1:n_eigen]
    Eigenfaces = t(train.scaled)%*%Eigenvectors
    
    projected.train = t(t(Eigenfaces)%*%t(train.scaled))
    projected.test = t(t(Eigenfaces)%*%t(test.scaled))
    
    #Fisher
    m = colMeans(projected.train)
    means.matrix = matrix(0,80,n_eigen)
    S.W = matrix(0,n_eigen,n_eigen)
    S.B = matrix(0,n_eigen,n_eigen)
    
    for(p in 1:80){
      p_th = seq(5*p-4,5*p,1) #pth person image indexes
      means.matrix[p,]= colMeans(projected.train[p_th,])
      S.W = (S.W) + (cov(projected.train[p_th,])*(length(p_th)-1))
      S.B = S.B + ((length(p_th))* (means.matrix[p,] - m)%*%t(means.matrix[p,] - m))
    }
    
    eig = eigen(solve(S.W)%*%S.B)
    eigenvectors.fisher = eig$vectors
    eigenvalues.fisher = eig$values
    
    train.new = Re(projected.train %*% eigenvectors.fisher[,1:79])
    test.new = Re(projected.test %*% eigenvectors.fisher[,1:79])

    for (k in ks){
      for (i_distance in distances) {
        predicted.label = matrix(0,80,1)
        for (m in 1:80) {
          predicted.label[m] = knn(train.new, train.labels, i_distance, test.new[m,], eigenvalues = eigenvalues.fisher, k = k, threshold = Inf) 
        }
        pred = sum(predicted.label== test.labels)/length(test.labels)
        accuracy = c(accuracy,pred)
      }
    }
  }  
}  
#24:12 min
length(accuracy)
distances.vector = rep(distances,126)
ks.vector = rep(ks,each = 7,42)
variances.vector = rep(variances,each = 21,6)
partition.vector = rep(partition,each = 147)

accuracy.df.knn.fisher = data.frame(distances.vector,ks.vector,variances.vector,partition.vector,accuracy)
names(accuracy.df.knn.fisher) = c("distances","k","variances","partition ","accuracy")
View(accuracy.df.knn.fisher)


acc.part.0 = accuracy.df.knn.fisher[which(accuracy.df.knn.fisher$`partition ` == 0),5]
acc.part.1 = accuracy.df.knn.fisher[which(accuracy.df.knn.fisher$`partition ` == 1),5]
acc.part.2 = accuracy.df.knn.fisher[which(accuracy.df.knn.fisher$`partition ` == 2),5]
acc.part.3 = accuracy.df.knn.fisher[which(accuracy.df.knn.fisher$`partition ` == 3),5]
acc.part.4 = accuracy.df.knn.fisher[which(accuracy.df.knn.fisher$`partition ` == 4),5]
acc.part.5 = accuracy.df.knn.fisher[which(accuracy.df.knn.fisher$`partition ` == 5),5]


matrix.acc.part = matrix(c(acc.part.0, acc.part.1, acc.part.2, acc.part.3, acc.part.4, acc.part.5),
                         length(acc.part.0),6)


mean.acc = apply(matrix.acc.part, 1,mean)

df.fish.knn.to.plot = accuracy.df.knn.fisher[1:147,1:3]

df.fish.knn.to.plot$accuracy = mean.acc


df.euclid = df.fish.knn.to.plot[which(df.fish.knn.to.plot$distances == "euclid"),]
df.modified_SSE = df.fish.knn.to.plot[which(df.fish.knn.to.plot$distances == "modified_SSE"),]
df.canberra = df.fish.knn.to.plot[which(df.fish.knn.to.plot$distances == "canberra"),]
df.simplified_mahalanobis = df.fish.knn.to.plot[which(df.fish.knn.to.plot$distances == "simplified_mahalanobis"),]
df.weighted_angle_based = df.fish.knn.to.plot[which(df.fish.knn.to.plot$distances == "weighted_angle_based"),]
df.angle_based = df.fish.knn.to.plot[which(df.fish.knn.to.plot$distances == "angle_based"),]
df.manhattan = df.fish.knn.to.plot[which(df.fish.knn.to.plot$distances == "manhattan"),]


{
  euclid = ggplot(df.euclid)+
    aes(x = variances, y = accuracy, col = as.character(k))+
    geom_line(size =1)+
    geom_point(size = 3)+
    ggtitle("Euclid")+
    labs(col = "k")
  
  modified_SSE = ggplot(df.modified_SSE)+
    aes(x = variances, y = accuracy, col = as.character(k))+
    geom_line(size =1)+
    geom_point(size = 3)+
    ggtitle("Modified SSE")
  
  canberra = ggplot(df.canberra)+
    aes(x = variances, y = accuracy, col = as.character(k))+
    geom_line(size =1)+
    geom_point(size = 3)+
    ggtitle("Canberra")
  
  simplified_mahalanobis = ggplot(df.simplified_mahalanobis)+
    aes(x = variances, y = accuracy, col = as.character(k))+
    geom_line(size =1)+
    geom_point(size = 3)+
    ggtitle("Simplified Mahalanobis")
  
  weighted_angle_based = ggplot(df.weighted_angle_based)+
    aes(x = variances, y = accuracy, col = as.character(k))+
    geom_line(size =1)+
    geom_point(size = 3)+
    ggtitle("Weighted Angle Based")
  
  angle_based = ggplot(df.angle_based)+
    aes(x = variances, y = accuracy, col = as.character(k))+
    geom_line(size =1)+
    geom_point(size = 3)+
    ggtitle("Angle Based")
  
  manhattan = ggplot(df.manhattan)+
    aes(x = variances, y = accuracy, col = as.character(k))+
    geom_line(size =1)+
    geom_point(size = 3)+
    ggtitle("Manhattan")
}

ggarrange(euclid, modified_SSE, canberra, simplified_mahalanobis, weighted_angle_based, angle_based, manhattan, ncol = 3,nrow = 3, common.legend = TRUE)+
  labs(col = "k")



View(df.fish.knn.to.plot[order(mean.acc, decreasing = T)[1:30],])#view the top 30 combination of hyperparameters that get the best results

table(df.fish.knn.to.plot[order(mean.acc, decreasing = T)[1:10],1]) 
table(df.fish.knn.to.plot[order(mean.acc, decreasing = T)[1:20],1]) 
table(df.fish.knn.to.plot[order(mean.acc, decreasing = T)[1:25],1]) 
table(df.fish.knn.to.plot[order(mean.acc, decreasing = T)[1:40],1]) 
table(df.fish.knn.to.plot[order(mean.acc, decreasing = T)[1:50],1]) 
#as we can see, the distance that get the best precision is the angle based followed by the modified SSE,
#after them, euclid and manhattan distance follow (but they follow by far)


modified_SSE_2 = ggplot(df.modified_SSE)+
  aes(x = variances, y = accuracy, col = as.character(k))+
  geom_line(size =1)+
  geom_point(size = 3)+
  ggtitle("Modified SSE")+
  labs(col = "k")+
  ylim(0.96, 1)

angle_based_2 = ggplot(df.angle_based)+
  aes(x = variances, y = accuracy, col = as.character(k))+
  geom_line(size =1)+
  geom_point(size = 3)+
  ggtitle("Angle Based")+
  ylim(0.96, 1)

ggarrange(modified_SSE_2, angle_based_2, ncol = 2,nrow = 1, common.legend = TRUE)+
  labs(col = "k")

#the angle based distance seems to be the best

ggplot(df.fish.knn.to.plot[order(mean.acc)[110:147],])+
  aes(x = 1:38, y = accuracy, col = df.fish.knn.to.plot[order(mean.acc)[110:147],]$distance)+
  geom_boxplot()

#we can conclude that the angle based is better for classification, but we will calculate the threshold with both 
#in order to see if it is easier to get it in the modified SSE


#threshold######
#first lets get the trainset that will be use for the training
  #pca
data_ = scale(data)
Sigma_ = data_%*%t(data_)/(nrow(data_)-1)

Eigen = eigen(Sigma_)
Eigenvalues = Eigen$values
Prop.Var = Eigenvalues/sum(Eigenvalues)
Cummulative.Var = cumsum(Eigenvalues)/sum(Eigenvalues)


n_eigen = 180
eigenvalues = Eigen$values[1:n_eigen]

Eigenvectors = Eigen$vectors[,1:n_eigen]
Eigenfaces = t(data_)%*%Eigenvectors
dim(data_)
dim(Eigenvectors)
pca_data = t(t(Eigenfaces)%*%t(data_))
dim(pca_data)
  #fisher

m = colMeans(pca_data)
means.matrix = matrix(0,80,n_eigen)
S.W = matrix(0,n_eigen,n_eigen)
S.B = matrix(0,n_eigen,n_eigen)

for(p in 1:80){
  p_th = seq(6*p-5,6*p,1) #pth person image indexes
  means.matrix[p,]= colMeans(pca_data[p_th,])
  S.W = (S.W) + (cov(pca_data[p_th,])*(length(p_th)-1))
  S.B = S.B + ((length(p_th))* (means.matrix[p,] - m)%*%t(means.matrix[p,] - m))
}

eig = eigen(solve(S.W)%*%S.B)
eigenvectors.fisher = eig$vectors
eigenvalues.fisher = eig$values

data.fisher = Re(pca_data %*% eigenvectors.fisher[,1:79])
dim(data.fisher)

#calculate the minimum distances between observations of the same class

fun.min.distances.within = function(data.fisher,distan){
  min.distances.within = vector()
  for(i in 1:80){
    i_th = seq(6*i-5,6*i,1) #pth person image indexes
    aux = combn(i_th, 2)
    aux2 = vector()
    for(j in 1:dim(aux)[2]){
       aux2 = c(aux2, distance(data.fisher[aux[1,j],],data.fisher[aux[2,j],],distan))
    }
    min.distances.within = c(min.distances.within, min(aux2))
  }
  return(min.distances.within)
}    

#angle based
min.distances.within.angle_based = fun.min.distances.within(data.fisher, "angle_based")

#modified SSE
min.distances.within.modified_SSE = fun.min.distances.within(data.fisher, "modified_SSE")


#calculate the minimum distances between every person and all the other images of different people in the training set

fun.min.distances.between = function(data.fisher,distan){
  index = 1:480
  min.distances.between = vector()
  for(i in 1:80){
    i_th = seq(6*i-5,6*i,1) #pth person image indexes
    
    for(j in i_th){
      aux = vector()
      for(k in index[-i_th]){
        aux = c(aux, distance(data.fisher[j,], data.fisher[k,],distan))
        }
      min.distances.between = c(min.distances.between, min(aux))  
      }
    }
  return(min.distances.between)
}    

min.distances.between.angle_based = fun.min.distances.between(data.fisher, "angle_based")

min.distances.between.modified_SSE = fun.min.distances.between(data.fisher, "modified_SSE")


  #for angle based
distances_angle_based = c(min.distances.within.angle_based,min.distances.between.angle_based)
type = c(rep("within same class",80), rep("between different classes",480))
angle_based_to_plot = data.frame(x = distances_angle_based, y = type)
f = sample(560)
plot.ang =  ggplot(angle_based_to_plot[f,])+
  aes(x = 1:560, y = distances_angle_based[f], col = type[f])+
  geom_point()+
  ggtitle("Min angle based distances")+
  labs(col = "")+
  xlab("")+
  ylab("distance")

#there is a huge space in which the threshold can be chosen


distances_modified_SSE = c(min.distances.within.modified_SSE,min.distances.between.modified_SSE)
type = c(rep("within same class",80), rep("between different classes",480))
modified_SSE_to_plot = data.frame(x = distances_modified_SSE, y = type)
f = sample(560)
ggplot(modified_SSE_to_plot)+
  aes(x = 1:560, y = distances_modified_SSE[f], col = type[f])+
  geom_point()+
  ggtitle("Min modified SSE distances")+
  labs(col = "")+
  xlab("")+
  ylab("distance")

#the results are not as good as with the angle based distance


min.limit = MinMaxObject(distances_angle_based)$min
max.limit = MinMaxObject(distances_angle_based)$max

walk.threshold = seq(min.limit-0.02,max.limit+0.02,0.001) 
length(walk.threshold)
thresholds.far = matrix(0,length(walk.threshold),1)

for(i in 1:length(walk.threshold)){
  thresholds.far[i] = sum(angle_based_to_plot[which(angle_based_to_plot$y == "between different classes"),1]<= walk.threshold[i])
}
thresholds.far = thresholds.far/length(angle_based_to_plot[which(angle_based_to_plot$y == "between different classes"),1])

thresholds.faf = matrix(0,length(walk.threshold),1)
for(i in 1:length(walk.threshold)){
  thresholds.faf[i] = sum(angle_based_to_plot[which(angle_based_to_plot$y == "within same class"),1]>= walk.threshold[i])
}
thresholds.faf = thresholds.faf / length(angle_based_to_plot[which(angle_based_to_plot$y == "within same class"),1])


thresholds.to.plot = data.frame(x =thresholds.far, y = thresholds.faf, walk.threshold)
ggplot(thresholds.to.plot)+
  aes(x = walk.threshold, y = 1, col = z)+
  geom_line(y = thresholds.to.plot[,1], col = "blue", show.legend = T)+
  geom_line(y = thresholds.to.plot[,2], col = "red", show.legend = T)+
  ggtitle("FAF (red) FAR (blue)")+
  xlab("threshold")+
  ylab("rate")+
  ylim(0,1)

#we will choose -0.59 as the threshold because althought most of the values of the min between diferent class is 
#closer to the threshold than the max of the distances within the same class, this might be beacause the are more observations of distances
#between classes, so the TRUE FAF might tend to the left







