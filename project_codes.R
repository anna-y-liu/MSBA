##images process
install.packages("BiocManager")
BiocManager::install("EBImage")

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
  df <- as.vector(images[[i]])
  image_df <- rbind(image_df,df)
}
rm(images,df)

for (i in 1:ncol(image_df)){
  colnames(image_df)[i] <- paste0("cube", i)
}
data <- cbind(image_df,concept[,2:5])


#linear regression
#set.seed(2)
train.rows <- sample(rownames(data), dim(data)[1]*0.7)
train.data <- data[train.rows, ]
valid.rows <- setdiff(rownames(data), train.rows)
valid.data <- data[valid.rows, ]
rm(i)

#glmnet linear regression
library(glmnet)
##TreeLabels
TreeLabels.fit <-  glmnet(as.matrix(train.data[,1:1200]), train.data$TreeLabels, type.measure="mse")
plot(TreeLabels.fit, xvar = "lambda", label = TRUE)
plot(TreeLabels.fit, xvar = "dev", label = TRUE)
TreeLabels.cv <-  cv.glmnet(as.matrix(train.data[,1:1200]), train.data$TreeLabels,nfolds = 20)
pre.lm <- predict(TreeLabels.fit, newx = as.matrix(valid.data[,1:1200]), type = 'response', s = TreeLabels.cv$lambda.min)
error <- valid.data$TreeLabels - pre.lm
TreeLabels.img.MSE <- sum(error ** 2) / length(error)
TreeLabels.img.MSE

##AnimalLabels
AnimalLabels.fit <-  glmnet(as.matrix(train.data[,1:1200]), train.data$AnimalLabels, type.measure="mse")
plot(AnimalLabels.fit, xvar = "lambda", label = TRUE)
plot(AnimalLabels.fit, xvar = "dev", label = TRUE)
AnimalLabels.cv <-  cv.glmnet(as.matrix(train.data[,1:1200]), train.data$AnimalLabels,nfolds = 20)
pre.lm <- predict(AnimalLabels.fit, newx = as.matrix(valid.data[,1:1200]), type = 'response', s = AnimalLabels.cv$lambda.min)
error <- valid.data$AnimalLabels - pre.lm
AnimalLabels.img.MSE <- sum(error ** 2) / length(error)
AnimalLabels.img.MSE

##MythologicalLabels_1
MythologicalLabels_1.fit <-  glmnet(as.matrix(train.data[,1:1200]), train.data$MythologicalLabels_1, type.measure="mse")
plot(MythologicalLabels_1.fit, xvar = "lambda", label = TRUE)
plot(MythologicalLabels_1.fit, xvar = "dev", label = TRUE)
MythologicalLabels_1.cv <-  cv.glmnet(as.matrix(train.data[,1:1200]), train.data$MythologicalLabels_1,nfolds = 20)
pre.lm <- predict(MythologicalLabels_1.fit, newx = as.matrix(valid.data[,1:1200]), type = 'response', s = MythologicalLabels_1.cv$lambda.min)
error <- valid.data$MythologicalLabels_1 - pre.lm
MythologicalLabels_1.img.MSE <- sum(error ** 2) / length(error)
MythologicalLabels_1.img.MSE

##MythologicalLabels_2
MythologicalLabels_2.fit <-  glmnet(as.matrix(train.data[,1:1200]), train.data$MythologicalLabels_2, type.measure="mse")
plot(MythologicalLabels_2.fit, xvar = "lambda", label = TRUE)
plot(MythologicalLabels_2.fit, xvar = "dev", label = TRUE)
MythologicalLabels_2.cv <-  cv.glmnet(as.matrix(train.data[,1:1200]), train.data$MythologicalLabels_2,nfolds = 20)
pre.lm <- predict(MythologicalLabels_2.fit, newx = as.matrix(valid.data[,1:1200]), type = 'response', s = MythologicalLabels_2.cv$lambda.min)
error <- valid.data$MythologicalLabels_2 - pre.lm
MythologicalLabels_2.img.MSE <- sum(error ** 2) / length(error)
MythologicalLabels_2.img.MSE

rm(AnimalLabels.cv,AnimalLabels.fit,
   TreeLabels.cv, TreeLabels.fit,
   MythologicalLabels_1.cv, MythologicalLabels_1.fit,
   MythologicalLabels_2.cv, MythologicalLabels_2.fit)

#text data
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
data <- data.frame(corpus, concept[,-1])
rownames(data) <- NULL
rm(corpus)

#linear regression
#set.seed(2)
train.data <- data[train.rows, ]
valid.data <- data[valid.rows, ]

#glmnet linear regression
library(glmnet)
##TreeLabels
TreeLabels.fit <-  glmnet(as.matrix(train.data[,1:1400]), train.data$TreeLabels, type.measure="mse")
plot(TreeLabels.fit, xvar = "lambda", label = TRUE)
plot(TreeLabels.fit, xvar = "dev", label = TRUE)
TreeLabels.cv <-  cv.glmnet(as.matrix(train.data[,1:1400]), train.data$TreeLabels,nfolds = 20)
pre.lm <- predict(TreeLabels.fit, newx = as.matrix(valid.data[,1:1400]), type = 'response', s = TreeLabels.cv$lambda.min)
error <- valid.data$TreeLabels - pre.lm
TreeLabels.text.MSE <- sum(error ** 2) / length(error)
TreeLabels.text.MSE

##AnimalLabels
AnimalLabels.fit <-  glmnet(as.matrix(train.data[,1:1400]), train.data$AnimalLabels, type.measure="mse")
plot(AnimalLabels.fit, xvar = "lambda", label = TRUE)
plot(AnimalLabels.fit, xvar = "dev", label = TRUE)
AnimalLabels.cv <-  cv.glmnet(as.matrix(train.data[,1:1400]), train.data$AnimalLabels,nfolds = 20)
pre.lm <- predict(AnimalLabels.fit, newx = as.matrix(valid.data[,1:1400]), type = 'response', s = AnimalLabels.cv$lambda.min)
error <- valid.data$AnimalLabels - pre.lm
AnimalLabels.text.MSE <- sum(error ** 2) / length(error)
AnimalLabels.text.MSE

##MythologicalLabels_1
MythologicalLabels_1.fit <-  glmnet(as.matrix(train.data[,1:1400]), train.data$MythologicalLabels_1, type.measure="mse")
plot(MythologicalLabels_1.fit, xvar = "lambda", label = TRUE)
plot(MythologicalLabels_1.fit, xvar = "dev", label = TRUE)
MythologicalLabels_1.cv <-  cv.glmnet(as.matrix(train.data[,1:1400]), train.data$MythologicalLabels_1,nfolds = 20)
pre.lm <- predict(MythologicalLabels_1.fit, newx = as.matrix(valid.data[,1:1400]), type = 'response', s = MythologicalLabels_1.cv$lambda.min)
error <- valid.data$MythologicalLabels_1 - pre.lm
MythologicalLabels_1.text.MSE <- sum(error ** 2) / length(error)
MythologicalLabels_1.text.MSE

##MythologicalLabels_2
MythologicalLabels_2.fit <-  glmnet(as.matrix(train.data[,1:1400]), train.data$MythologicalLabels_2, type.measure="mse")
plot(MythologicalLabels_2.fit, xvar = "lambda", label = TRUE)
plot(MythologicalLabels_2.fit, xvar = "dev", label = TRUE)
MythologicalLabels_2.cv <-  cv.glmnet(as.matrix(train.data[,1:1400]), train.data$MythologicalLabels_2,nfolds = 20)
pre.lm <- predict(MythologicalLabels_2.fit, newx = as.matrix(valid.data[,1:1400]), type = 'response', s = MythologicalLabels_2.cv$lambda.min)
error <- valid.data$MythologicalLabels_2 - pre.lm
MythologicalLabels_2.text.MSE <- sum(error ** 2) / length(error)
MythologicalLabels_2.text.MSE

rm(AnimalLabels.cv,AnimalLabels.fit,
   TreeLabels.cv, TreeLabels.fit,
   MythologicalLabels_1.cv, MythologicalLabels_1.fit,
   MythologicalLabels_2.cv, MythologicalLabels_2.fit)

#Image and text
data <- data.frame(image_df,data)

#linear regression
#set.seed(2)
train.data <- data[train.rows, ]
valid.data <- data[valid.rows, ]

#glmnet linear regression
library(glmnet)
##TreeLabels
TreeLabels.fit <-  glmnet(as.matrix(train.data[,1:2600]), train.data$TreeLabels, type.measure="mse")
plot(TreeLabels.fit, xvar = "lambda", label = TRUE)
plot(TreeLabels.fit, xvar = "dev", label = TRUE)
TreeLabels.cv <-  cv.glmnet(as.matrix(train.data[,1:2600]), train.data$TreeLabels,nfolds = 20)
pre.lm <- predict(TreeLabels.fit, newx = as.matrix(valid.data[,1:2600]), type = 'response', s = TreeLabels.cv$lambda.min)
error <- valid.data$TreeLabels - pre.lm
TreeLabels.iandt.MSE <- sum(error ** 2) / length(error)
TreeLabels.iandt.MSE

##AnimalLabels
AnimalLabels.fit <-  glmnet(as.matrix(train.data[,1:2600]), train.data$AnimalLabels, type.measure="mse")
plot(AnimalLabels.fit, xvar = "lambda", label = TRUE)
plot(AnimalLabels.fit, xvar = "dev", label = TRUE)
AnimalLabels.cv <-  cv.glmnet(as.matrix(train.data[,1:2600]), train.data$AnimalLabels,nfolds = 20)
pre.lm <- predict(AnimalLabels.fit, newx = as.matrix(valid.data[,1:2600]), type = 'response', s = AnimalLabels.cv$lambda.min)
error <- valid.data$AnimalLabels - pre.lm
AnimalLabels.iandt.MSE <- sum(error ** 2) / length(error)
AnimalLabels.iandt.MSE

##MythologicalLabels_1
MythologicalLabels_1.fit <-  glmnet(as.matrix(train.data[,1:2600]), train.data$MythologicalLabels_1, type.measure="mse")
plot(MythologicalLabels_1.fit, xvar = "lambda", label = TRUE)
plot(MythologicalLabels_1.fit, xvar = "dev", label = TRUE)
MythologicalLabels_1.cv <-  cv.glmnet(as.matrix(train.data[,1:2600]), train.data$MythologicalLabels_1,nfolds = 20)
pre.lm <- predict(MythologicalLabels_1.fit, newx = as.matrix(valid.data[,1:2600]), type = 'response', s = MythologicalLabels_1.cv$lambda.min)
error <- valid.data$MythologicalLabels_1 - pre.lm
MythologicalLabels_1.iandt.MSE <- sum(error ** 2) / length(error)
MythologicalLabels_1.iandt.MSE

##MythologicalLabels_2
MythologicalLabels_2.fit <-  glmnet(as.matrix(train.data[,1:2600]), train.data$MythologicalLabels_2, type.measure="mse")
plot(MythologicalLabels_2.fit, xvar = "lambda", label = TRUE)
plot(MythologicalLabels_2.fit, xvar = "dev", label = TRUE)
MythologicalLabels_2.cv <-  cv.glmnet(as.matrix(train.data[,1:2600]), train.data$MythologicalLabels_2,nfolds = 20)
pre.lm <- predict(MythologicalLabels_2.fit, newx = as.matrix(valid.data[,1:2600]), type = 'response', s = MythologicalLabels_2.cv$lambda.min)
error <- valid.data$MythologicalLabels_2 - pre.lm
MythologicalLabels_2.iandt.MSE <- sum(error ** 2) / length(error)
MythologicalLabels_2.iandt.MSE

rm(AnimalLabels.cv,AnimalLabels.fit,
   TreeLabels.cv, TreeLabels.fit,
   MythologicalLabels_1.cv, MythologicalLabels_1.fit,
   MythologicalLabels_2.cv, MythologicalLabels_2.fit)

data.frame(pre.lm,valid.data$TreeLabels)
prediction <- ifelse(pre.lm >0.85,1,0)

library(caret)
confusionMatrix(as.factor(prediction), as.factor(valid.data$TreeLabels))

