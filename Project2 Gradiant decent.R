#load data
# ##images process
# install.packages("BiocManager")
# BiocManager::install("EBImage")
library(EBImage)
#read concept
library(readr)
TreeLabels <- read.csv("Project/ProjectData/TreeLabels.csv")
TreeLabels <- TreeLabels[,-c(2:8)]
colnames(TreeLabels)[2] <- "TreeLabels"
colnames(TreeLabels)[1] <- "Image"
AnimalLabels <- read.csv("Project/ProjectData/AnimalLabels.csv")
AnimalLabels <- AnimalLabels[,-c(2:8)]
colnames(AnimalLabels)[1] <- "Image"
colnames(AnimalLabels)[2] <- "AnimalLabels"
MythologicalLabels <- read.csv("Project/ProjectData/MythologicalLabels.csv")
MythologicalLabels <- MythologicalLabels[,-c(2:8)]
colnames(MythologicalLabels)[2] <- "MythologicalLabels_1"
colnames(MythologicalLabels)[3] <- "MythologicalLabels_2"
colnames(MythologicalLabels)[1] <- "Image"
concept <- merge(TreeLabels,AnimalLabels)
concept <- merge(concept,MythologicalLabels)
rm(TreeLabels,AnimalLabels,MythologicalLabels)

#read images
list<-list.files(path = "Project/ProjectData/s1/img/", pattern = ".jpg",full.names = TRUE)
list<-append(list,list.files(path = "Project/ProjectData/s2/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "Project/ProjectData/s3/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "Project/ProjectData/s4/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "Project/ProjectData/s5/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "Project/ProjectData/s6/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "Project/ProjectData/s7/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "Project/ProjectData/s8/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "Project/ProjectData/s9/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "Project/ProjectData/s10/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "Project/ProjectData/s11/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "Project/ProjectData/s12/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "Project/ProjectData/s13/img/", pattern = ".jpg",full.names = TRUE))
list_of_images <- lapply(list,readImage)
images <- lapply(list_of_images, resize,w=20,h=20)
rm(list_of_images,list)

#flatten image data
image_df <- data.frame(matrix(ncol = 12))
image_df <- image_df[-1,]
for (i in 1:106){
  df <- as.vector(images[[i]][,,1]+images[[i]][,,2]+images[[i]][,,3])
  image_df <- rbind(image_df,df)
}
rm(images,df)

for (i in 1:ncol(image_df)){
  colnames(image_df)[i] <- paste0("cube", i)
}

#Text data
library(tm)
library(dplyr)
library(stringr)

removeCommonTerms <- function(X, percentage) {
  x = t(X)
  t = table(x$i) < x$ncol * percentage
  X[, as.numeric(names(t[t]))]
}

corpus = Corpus(DirSource("Project/ProjectData/text data", encoding = "UTF-8", recursive = TRUE, ignore.case = FALSE, mode = "text"),
) %>%
  tm_map(content_transformer(tolower)) %>%        # no uppercase
  tm_map(removeWords, stopwords('en')) %>%        # remove stopwords
  tm_map(removePunctuation) %>%                   # no punctuation
  tm_map(stripWhitespace) %>%                     # no extra whitespaces
  tm_map(stemDocument) %>%                        # reduce to radical
  DocumentTermMatrix(
    control = list(
      weighting = weightTf,
      wordLengths = c(3,30),                  # radical between 3 and 30
      minDocFreq = 1                          # appears at least 1 times
    )
    
  ) %>%
  removeCommonTerms(0.70) %>%                     # maximum 70% documents
  as.matrix()
summary(t(data.frame(lapply(data.frame(corpus), function(x) sum(x==0) / length(x)))))

#convert to dataframe
image_text_df <- cbind(image_df, corpus, concept[,-1])
image_text_df <- image_text_df[lapply(image_text_df, function(x) sum(x==0) / length(x) ) < 0.90]
rownames(image_text_df) <- NULL
image_df <- cbind(image_df,concept[,-1])
text_df <- data.frame(corpus, concept[,-1]) 
text_df <- text_df[lapply(text_df, function(x) sum(x==0) / length(x) ) < 0.90]
rownames(text_df) <- NULL
rm(corpus)

# You are required to test code to implement ANY TWO of the following algorithms: 
# (a) Quasi Newton Method – either using Eigen Decomposition or Cholesky Factorization
# (b) Heavy Ball Method (also called Momentum Method) or Nesterov’s Method and 
# (c) Variants of Gradient Descent such as ADAGrad, ADADelta or RMSProp and 
# (d) Conjugate Gradient Method.
concept_name <- c("TreeLabels", "AnimalLabels", 
                         "MythologicalLabels_1", "MythologicalLabels_2")

#trace("MGD", edit = TRUE)

#Tree
#image and text
library(gradDescent)
featureScalingResult <- varianceScaling(image_text_df[,1:446])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
run_time <- Sys.time()
model <- MGD(splitedDataSet$dataTrain, alpha=0.01, maxIter=200, momentum=0.8)
run_time <- Sys.time() - run_time
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[446] - prediction[446]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[446] - prediction[446]
MSE <- sum(error ** 2) / length(error)
MSE

#ADADelta
featureScalingResult <- varianceScaling(image_text_df[,1:446])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
run_time <- Sys.time()
model <- ADADELTA(splitedDataSet$dataTrain, maxIter=3000, momentum=0.9)
run_time <- Sys.time() - run_time
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[446] - prediction[446]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[446] - prediction[446]
MSE <- sum(error ** 2) / length(error)
MSE

#Image only
featureScalingResult <- varianceScaling(image_df[,1:401])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
model <- MGD(splitedDataSet$dataTrain, alpha=0.01, maxIter=200, momentum=0.9)
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[401] - prediction[401]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[401] - prediction[401]
MSE <- sum(error ** 2) / length(error)
MSE

#ADADelta
featureScalingResult <- varianceScaling(image_df[,1:401])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
run_time <- Sys.time()
model <- ADADELTA(splitedDataSet$dataTrain, maxIter=3000, momentum=0.9)
run_time <- Sys.time() - run_time
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[401] - prediction[401]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[401] - prediction[401]
MSE <- sum(error ** 2) / length(error)
MSE

#With text only
featureScalingResult <- varianceScaling(text_df[,1:46])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
model <- MGD(splitedDataSet$dataTrain, alpha=0.01, maxIter=200, momentum=0.9)
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[46] - prediction[46]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[46] - prediction[46]
MSE <- sum(error ** 2) / length(error)
MSE

#ADADelta
featureScalingResult <- varianceScaling(text_df[,1:46])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
run_time <- Sys.time()
model <- ADADELTA(splitedDataSet$dataTrain, maxIter=3000, momentum=0.9)
run_time <- Sys.time() - run_time
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[46] - prediction[46]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[46] - prediction[46]
MSE <- sum(error ** 2) / length(error)
MSE

#Animal
#image and text
featureScalingResult <- varianceScaling(image_text_df[,c(1:445,447)])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
model <- MGD(splitedDataSet$dataTrain, alpha=0.01, maxIter=200, momentum=0.9)
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[446] - prediction[446]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[446] - prediction[446]
MSE <- sum(error ** 2) / length(error)
MSE

#ADADelta
featureScalingResult <- varianceScaling(image_text_df[,c(1:445,447)])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
run_time <- Sys.time()
model <- ADADELTA(splitedDataSet$dataTrain, maxIter=3000, momentum=0.9)
run_time <- Sys.time() - run_time
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[446] - prediction[446]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[446] - prediction[446]
MSE <- sum(error ** 2) / length(error)
MSE

#Image only
featureScalingResult <- varianceScaling(image_df[,1:401])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
model <- MGD(splitedDataSet$dataTrain, alpha=0.01, maxIter=200, momentum=0.9)
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[401] - prediction[401]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[401] - prediction[401]
MSE <- sum(error ** 2) / length(error)
MSE

#ADADelta
featureScalingResult <- varianceScaling(image_df[,1:401])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
run_time <- Sys.time()
model <- ADADELTA(splitedDataSet$dataTrain, maxIter=3000, momentum=0.9)
run_time <- Sys.time() - run_time
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[401] - prediction[401]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[401] - prediction[401]
MSE <- sum(error ** 2) / length(error)
MSE

#With text only
featureScalingResult <- varianceScaling(text_df[,1:46])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
model <- MGD(splitedDataSet$dataTrain, alpha=0.01, maxIter=200, momentum=0.9)
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[46] - prediction[46]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[46] - prediction[46]
MSE <- sum(error ** 2) / length(error)
MSE

#ADADelta
featureScalingResult <- varianceScaling(text_df[,1:46])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
run_time <- Sys.time()
model <- ADADELTA(splitedDataSet$dataTrain, maxIter=3000, momentum=0.9)
run_time <- Sys.time() - run_time
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[46] - prediction[46]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[46] - prediction[46]
MSE <- sum(error ** 2) / length(error)
MSE

#Myth_1
#image and text
featureScalingResult <- varianceScaling(image_text_df[,c(1:445,448)])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
model <- MGD(splitedDataSet$dataTrain, alpha=0.01, maxIter=200, momentum=0.9)
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[446] - prediction[446]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[446] - prediction[446]
MSE <- sum(error ** 2) / length(error)
MSE

#ADADelta
featureScalingResult <- varianceScaling(image_text_df[,c(1:445,448)])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
run_time <- Sys.time() 
model <- ADADELTA(splitedDataSet$dataTrain, maxIter=3000, momentum=0.9) 
run_time <- Sys.time() - run_time
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[446] - prediction[446]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[446] - prediction[446]
MSE <- sum(error ** 2) / length(error)
MSE

#Image only
featureScalingResult <- varianceScaling(image_df[,1:401])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
model <- MGD(splitedDataSet$dataTrain, alpha=0.01, maxIter=200, momentum=0.9)
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[401] - prediction[401]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[401] - prediction[401]
MSE <- sum(error ** 2) / length(error)
MSE

#ADADelta
featureScalingResult <- varianceScaling(image_df[,1:401])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
run_time <- Sys.time() 
model <- ADADELTA(splitedDataSet$dataTrain, maxIter=3000, momentum=0.9) 
run_time <- Sys.time() - run_time
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[401] - prediction[401]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[401] - prediction[401]
MSE <- sum(error ** 2) / length(error)
MSE

#With text only
featureScalingResult <- varianceScaling(text_df[,1:46])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
model <- MGD(splitedDataSet$dataTrain, alpha=0.01, maxIter=200, momentum=0.9)
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[46] - prediction[46]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[46] - prediction[46]
MSE <- sum(error ** 2) / length(error)
MSE

#ADADelta
featureScalingResult <- varianceScaling(text_df[,1:46])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
run_time <- Sys.time() 
model <- ADADELTA(splitedDataSet$dataTrain, maxIter=3000, momentum=0.9) 
run_time <- Sys.time() - run_time
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[46] - prediction[46]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[46] - prediction[46]
MSE <- sum(error ** 2) / length(error)
MSE


#Myth_2
#image and text
featureScalingResult <- varianceScaling(image_text_df[,c(1:445,449)])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
model <- MGD(splitedDataSet$dataTrain, alpha=0.01, maxIter=200, momentum=0.9)
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[446] - prediction[446]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[446] - prediction[446]
MSE <- sum(error ** 2) / length(error)
MSE

#ADADelta
featureScalingResult <- varianceScaling(image_text_df[,c(1:445,449)])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
run_time <- Sys.time() 
model <- ADADELTA(splitedDataSet$dataTrain, maxIter=3000, momentum=0.9) 
run_time <- Sys.time() - run_time
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[446] - prediction[446]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[446] - prediction[446]
MSE <- sum(error ** 2) / length(error)
MSE

#Image only
featureScalingResult <- varianceScaling(image_df[,1:401])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
model <- MGD(splitedDataSet$dataTrain, alpha=0.01, maxIter=200, momentum=0.9)
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[401] - prediction[401]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[401] - prediction[401]
MSE <- sum(error ** 2) / length(error)
MSE

#ADADelta
featureScalingResult <- varianceScaling(image_df[,1:401])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
run_time <- Sys.time() 
model <- ADADELTA(splitedDataSet$dataTrain, maxIter=3000, momentum=0.9) 
run_time <- Sys.time() - run_time
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[401] - prediction[401]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[401] - prediction[401]
MSE <- sum(error ** 2) / length(error)
MSE

#With text only
featureScalingResult <- varianceScaling(text_df[,1:46])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
model <- MGD(splitedDataSet$dataTrain, alpha=0.01, maxIter=200, momentum=0.9)
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[46] - prediction[46]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[46] - prediction[46]
MSE <- sum(error ** 2) / length(error)
MSE

#ADADelta
featureScalingResult <- varianceScaling(text_df[,1:46])
splitedDataSet <- splitData(featureScalingResult$scaledDataSet, dataTrainRate = 0.8)
run_time <- Sys.time() 
model <- ADADELTA(splitedDataSet$dataTrain, maxIter=3000, momentum=0.9) 
run_time <- Sys.time() - run_time
## separate testing data with input only
dataTestInput <- (splitedDataSet$dataTest)[,1:ncol(splitedDataSet$dataTest)-1]
## predict testing data using GD model
prediction <- prediction(model,dataTestInput)
error <- (splitedDataSet$dataTest)[46] - prediction[46]
MSE <- sum(error ** 2) / length(error)
MSE

##predict with all dataset
dataTestInput <- featureScalingResult$scaledDataSet[,1:ncol(splitedDataSet$dataTest)-1]
prediction <- prediction(model,dataTestInput)
error <- featureScalingResult$scaledDataSet[46] - prediction[46]
MSE <- sum(error ** 2) / length(error)
MSE
