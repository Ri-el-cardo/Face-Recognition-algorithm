library(bmp)
library(OpenImageR)
library(ggplot2)
library(ggpubr)
library(e1071)

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
  
  
  #this generates indexes in which every 6 indexes(each group corresponds to the same person in the trainset) are randomly rearranged
  rearranged.indexes = as.vector(replicate(80,sample(6))) + sapply(0:479, function(i)floor(i/6)*6) 
  
  #rearranged.indexes[seq(1,480,6)+i]
  #when i ranges from 0:5, those five sets are a partition of the dataset in which there is 1 observation of each person
  
}  

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
    
    
    
    if (k != "linear"){
      for (g in gamma){
        
        
        
        classifier=svm(formula=train.labels ~. ,data=train.new,
                       type='C-classification',
                       cost=.Machine$double.xmax,
                       kernel=k,
                       gamma = g,
                       scale = F)
        
        
        predict_v = predict(classifier, newdata = test.new)
        confusion = table(Real = test.labels, Pred = predict_v)
        model_accuracy = (sum(diag(confusion)))/(sum(confusion))
        accuracy <- c(accuracy, model_accuracy)
        
      }
    }
    
    if (k == "linear"){
      classifier=svm(formula=train.labels ~. ,data=train.new,
                     type='C-classification',
                     cost=.Machine$double.xmax,
                     kernel=k,
                     scale = F)
      
      
      predict_v = predict(classifier, newdata = test.new)
      confusion = table(Real = test.labels, Pred = predict_v)
      model_accuracy = (sum(diag(confusion)))/(sum(confusion))
      accuracy <- c(accuracy, model_accuracy)
    }
  }
}  

accuracy.linear = accuracy
length(accuracy.linear)
vect.lin.acc.var = rep(variances,6)
vect.lin.acc.part = rep(partition,each = 7)
vect.lin.acc.type = rep("linear",42)
length(vect.lin.acc.part)

df.acc.linear = data.frame(variance = vect.lin.acc.var, partition = vect.lin.acc.part, kernel = vect.lin.acc.type, accuracy = accuracy.linear)


acc.part.0 = df.acc.linear[which(df.acc.linear$partition  == 0),4]
acc.part.1 = df.acc.linear[which(df.acc.linear$partition  == 1),4]
acc.part.2 = df.acc.linear[which(df.acc.linear$partition  == 2),4]
acc.part.3 = df.acc.linear[which(df.acc.linear$partition  == 3),4]
acc.part.4 = df.acc.linear[which(df.acc.linear$partition  == 4),4]
acc.part.5 = df.acc.linear[which(df.acc.linear$partition  == 5),4]


matrix.acc.part = matrix(c(acc.part.0, acc.part.1, acc.part.2, acc.part.3, acc.part.4, acc.part.5),
                         length(acc.part.0),7)


mean.acc = apply(matrix.acc.part, 1,mean)

df.to.plot = df.acc.linear[1:7,1:3]


    