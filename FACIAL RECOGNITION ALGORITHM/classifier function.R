#here are some functions that we will use in the classifier 
{
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
  
classifier =  function(train, train.labels, test){
  data.separated = normalize_data(train,test)
  
  train.scaled = data.separated$train.scaled
  test.scaled = data.separated$test.scaled
  
  Sigma_ = train.scaled%*%t(train.scaled)/(nrow(train.scaled)-1)
  
  Eigen = eigen(Sigma_)
  Eigenvalues = Eigen$values
  Prop.Var = Eigenvalues/sum(Eigenvalues)
  
  Cummulative.Var = cumsum(Eigenvalues)/sum(Eigenvalues)
  
  n_eigen = 180
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
    p_th = seq(6*p-5,6*p,1) #pth person image indexes
    means.matrix[p,]= colMeans(projected.train[p_th,])
    S.W = (S.W) + (cov(projected.train[p_th,])*(length(p_th)-1))
    S.B = S.B + ((length(p_th))* (means.matrix[p,] - m)%*%t(means.matrix[p,] - m))
  }
  
  eig = eigen(solve(S.W)%*%S.B)
  eigenvectors.fisher = eig$vectors
  eigenvalues.fisher = eig$values
 
  train.new = Re(projected.train %*% eigenvectors.fisher[,1:79]) 
  test.new = Re(projected.test %*% eigenvectors.fisher[,1:79])
  predicted.label = matrix(0,nrow(test.new) ,1)
  for (m in 1:nrow(test.new)) {
    predicted.label[m] = knn(train.new, train.labels, "angle_based", test.new[m,], eigenvalues = eigenvalues.fisher, k = 1, threshold = -0.59) 
  }
  return(predicted.label)
}
