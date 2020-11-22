library(bmp)
library(OpenImageR)
library(ggplot2)
library(ggpubr)
setwd("E:/UNIVERSIDAD/3. Ciencia e ingeniería de datos/2º/1er semestre/Statistical Learning/eigenfaces/train")

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

#generates a dataframe in which the variables are the partitions of the dataset and the hyperparameters 
#in order to add later the accuracy of the model with every combination of this variables 
{
partition = c(0,1,2,3,4,5)
variances = c(0.9,0.91,0.92,0.93,0.94,0.95,0.96)
ks = c(1,2,3)
distances = c("euclid","modified_SSE","canberra",
              "simplified_mahalanobis","weighted_angle_based","angle_based",
              "manhattan")

distances.vector = rep(distances,126)
ks.vector = rep(ks,each = 7,42)
variances.vector = rep(variances,each = 21,6)
partition.vector = rep(partition,each = 147)

accuracy.df = data.frame(distances.vector,ks.vector,variances.vector,partition.vector)
names(accuracy.df) = c("distances","k","variances","partition ")
}

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
{distances = c("euclid","modified_SSE","canberra",
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
    
    for (k in ks){
      for (l in distances) {
        predicted.label = matrix(0,80,1)
        for (m in 1:80) {
          predicted.label[m] = knn(projected.train, train.labels, l, projected.test[m,], eigenvalues = eigenvalues, k = k, threshold = 100000000000000); 
          
        }
        pred = sum(predicted.label== test.labels)/length(test.labels)
        accuracy = c(accuracy,pred)
      }
    }
  }  
}  

accuracy.df$accuracy = accuracy 

#mean of the accuracy with the same hyperparameters, this is done because with each combiantion of hyperparameters 
#there are 6 observations corresponding with the accuracy because we divide the dataset in 6 partitions and then used
#the k-folds crossvalidation


#this dataframe contains uniquely every combination of the hyperparameters
df = accuracy.df[1:75,1:3]


acc.part.0 = accuracy.df[which(accuracy.df$`partition ` == 0),5]
acc.part.1 = accuracy.df[which(accuracy.df$`partition ` == 1),5]
acc.part.2 = accuracy.df[which(accuracy.df$`partition ` == 2),5]
acc.part.3 = accuracy.df[which(accuracy.df$`partition ` == 3),5]
acc.part.4 = accuracy.df[which(accuracy.df$`partition ` == 4),5]
acc.part.5 = accuracy.df[which(accuracy.df$`partition ` == 5),5]

matrix.acc.part = matrix(c(acc.part.0, acc.part.1, acc.part.2, acc.part.3, acc.part.4, acc.part.5),
                         length(acc.part.0),6)

mean.acc = apply(matrix.acc.part, 1,mean)

df$accuracy = mean.acc

df.euclid = df[which(df$distances == "euclid"),]
df.modified_SSE = df[which(df$distances == "modified_SSE"),]
df.canberra = df[which(df$distances == "canberra"),]
df.simplified_mahalanobis = df[which(df$distances == "simplified_mahalanobis"),]
df.weighted_angle_based = df[which(df$distances == "weighted_angle_based"),]

#these are the diferent accuracy plots for every combination of the hyperparameters
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
}

ggarrange(euclid,modified_SSE,canberra,simplified_mahalanobis,weighted_angle_based, ncol = 2,nrow = 3, common.legend = TRUE)+
  labs(col = "k")


#this plots allows us to choose which are the best combination of hyperparameters
#we can see that the canberra distance is the best one that we can 


#now it is time to choose the threshold
#matrix of indexes of the data in which each column contains the indexes of observations of the same class
aux.separation =  matrix(1:480,6,80)

#scale the data
data.scaled = scale(data, center = T, scale = T)

Sigma_ = data.scaled%*%t(data.scaled)/(nrow(data.scaled)-1)

Eigen = eigen(Sigma_)
Eigenvalues = Eigen$values
Prop.Var = Eigenvalues/sum(Eigenvalues)
Cummulative.Var = cumsum(Eigenvalues)/sum(Eigenvalues)

min(which(Cummulative.Var>0.95))

Eigenvectors = Eigen$vectors[,1:173]
Eigenfaces = t(data.scaled)%*%Eigenvectors

data.new = t(t(Eigenfaces)%*%t(data.scaled))

#return the maximum and the minimum distance within the obserbations of the same class (if every class has 6 observations)
distances.between.max = function(data.class){
  distances.list = matrix(0,15,1)
  
  distances.list[1] = distance(data.class[6,],data.class[1,],"canberra")
  distances.list[2] = distance(data.class[6,],data.class[2,],"canberra")
  distances.list[3] = distance(data.class[6,],data.class[3,],"canberra")
  distances.list[4] = distance(data.class[6,],data.class[4,],"canberra")
  distances.list[5] = distance(data.class[6,],data.class[5,],"canberra")
  distances.list[6] = distance(data.class[1,],data.class[2,],"canberra")
  distances.list[7] = distance(data.class[1,],data.class[3,],"canberra")
  distances.list[8] = distance(data.class[1,],data.class[4,],"canberra")
  distances.list[9] = distance(data.class[1,],data.class[5,],"canberra")
  distances.list[10] = distance(data.class[2,],data.class[3,],"canberra")
  distances.list[11] = distance(data.class[2,],data.class[4,],"canberra")
  distances.list[12] = distance(data.class[2,],data.class[5,],"canberra")
  distances.list[13] = distance(data.class[3,],data.class[4,],"canberra")
  distances.list[14] = distance(data.class[3,],data.class[5,],"canberra")
  distances.list[15] = distance(data.class[4,],data.class[5,],"canberra")
  
  return(max(distances.list))
}

distances.between.min = function(data.class){
  distances.list = matrix(0,15,1)
  
  distances.list[1] = distance(data.class[6,],data.class[1,],"canberra")
  distances.list[2] = distance(data.class[6,],data.class[2,],"canberra")
  distances.list[3] = distance(data.class[6,],data.class[3,],"canberra")
  distances.list[4] = distance(data.class[6,],data.class[4,],"canberra")
  distances.list[5] = distance(data.class[6,],data.class[5,],"canberra")
  distances.list[6] = distance(data.class[1,],data.class[2,],"canberra")
  distances.list[7] = distance(data.class[1,],data.class[3,],"canberra")
  distances.list[8] = distance(data.class[1,],data.class[4,],"canberra")
  distances.list[9] = distance(data.class[1,],data.class[5,],"canberra")
  distances.list[10] = distance(data.class[2,],data.class[3,],"canberra")
  distances.list[11] = distance(data.class[2,],data.class[4,],"canberra")
  distances.list[12] = distance(data.class[2,],data.class[5,],"canberra")
  distances.list[13] = distance(data.class[3,],data.class[4,],"canberra")
  distances.list[14] = distance(data.class[3,],data.class[5,],"canberra")
  distances.list[15] = distance(data.class[4,],data.class[5,],"canberra")
  
  return(min(distances.list))
}

list.min.distances.between = matrix(0,80,1)

for (i in 1:80) {
  c = data.new[aux.separation[,i],]
  list.min.distances.between[i] = distances.between.min(c)
}

list.max.distances.between = matrix(0,80,1)

for (i in 1:80) {
  c = data.new[aux.separation[,i],]
  list.max.distances.between[i] = distances.between.max(c)
}

min(list.max.distances.between)
max(list.min.distances.between)

ordered.min = list.min.distances.between[order(list.min.distances.between)]

ggplot()+aes(x =1:length(ordered.min) , y = ordered.min, col = "")+
  geom_point()+
  labs(title = "Ordered Min Distance Within Classes",y = "distance",x = "oredered classes")+
  theme(legend.position='none')+
  geom_abline(slope = 0, intercept = 103)

ggplot()+aes(x =1:length(ordered.min) , y = ordered.min, col = "")+
  geom_point()+
  labs(title = "Ordered Min Distance Within Classes",y = "distance",x = "oredered classes")+
  theme(legend.position='none')+
  geom_abline(slope = 0, intercept = 105)


all.comb.min = matrix(0,480,480) 
print(all.comb.min)
ueue = 1
for(i in 1:80){
  for (j in as.vector(aux.separation[,-i]) ) {
    for (k in 1:6) {
      n = ((i-1)*6)+k
      all.comb.min[n,j] = distance(data.new[j,],data.new[n,],"canberra")
    }
  }
}

all.comb.min[which(all.comb.min>1)]
min(all.comb.min[which(all.comb.min>1)])


ordered.min.all.comb = all.comb.min[order(all.comb.min[which(all.comb.min>1)])]

ggplot()+aes(x =1:length(ordered.min.all.comb) , y = ordered.min.all.comb, col = "")+
  geom_jitter(size = 0.5)+
  labs(title = "Distance Between Classes",y = "distance",x = "pairs of observations")+
  theme(legend.position='none')+
  geom_abline(slope = 0, intercept = 103)


ggplot()+aes(x =1:length(ordered.min.all.comb) , y = ordered.min.all.comb, col = "")+
  geom_jitter(size = 0.5)+
  labs(title = "Distance Between Classes",y = "distance",x = "pairs of observations")+
  theme(legend.position='none')+
  geom_abline(slope = 0, intercept = 105)
