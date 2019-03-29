rm(list=ls())
setwd("C:/Users/hp/Desktop/minor-harsh/images")
library("keras")
dataset = read.csv("Imagedataset2.csv")

library(caTools)
set.seed(123)

sam = read.csv('test.csv')
labels = sam$X
labels = as.vector(labels)
dataset= cbind(dataset,labels)
dataset$labels = factor(dataset$labels,
                        levels = c('Hernia','Pneumonia','Fibrosis','Edema','Emphysema',
                                   'Cardiomegaly','Pleural_Thickening','Consolidation',
                                   'Pneumothorax','Mass','Nodule','Atelectasis',
                                   'Effusion','Infiltration','No Finding'),
                        labels = c(0,1, 2, 3,4,5,6,7,8,9,10,11,12,13,14))


split = sample.split(dataset$labels, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

X_train = as.matrix(training_set[-16385])
Y_train = unlist(training_set[16385])
Y_train=as.vector(Y_train)

X_test = as.matrix(test_set[-16385])
Y_test = unlist(test_set[16385])
Y_test=as.vector(Y_test)

dim(X_test)=c(nrow(X_test),128,128,1)
dim(X_train)=c(nrow(X_train),128,128,1)

Y_train = to_categorical(Y_train, num_classes = 15)

Y_test = to_categorical(Y_test, num_classes = 15)

model <- keras_model_sequential()

model%>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), strides = c(1L, 1L),activation = 'relu', 
                input_shape = c(128,128,1)) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), strides = c(1L, 1L),activation = 'relu' )%>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.20) %>%
  
  layer_conv_2d(filters = 128, kernel_size = c(3,3),strides = c(1L, 1L), activation = 'relu')  %>%
  layer_conv_2d(filters = 128, kernel_size = c(3,3),strides = c(1L, 1L), activation = 'relu')  %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.20) %>%
  
  layer_conv_2d(filters = 86, kernel_size = c(3,3),strides = c(1L, 1L), activation = 'relu')  %>%
  layer_conv_2d(filters = 86, kernel_size = c(3,3),strides = c(1L, 1L), activation = 'relu')  %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  layer_conv_2d(filters = 86, kernel_size = c(3,3),strides = c(1L, 1L), activation = 'relu')  %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  
  
  layer_flatten() %>% 
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 15, activation = 'softmax')%>%
  
  compile(
    loss = 'categorical_crossentropy', 
    optimizer = optimizer_adam(),
    metrics=c("accuracy"))

history = model %>% fit(X_train, Y_train, epochs = 5, batch_size = 20,validation_split = 0.2)


setwd("C:/Users/hp/Desktop/minor-harsh/server")

save_model_hdf5(model,"mymodel.h5")


gen_images = image_data_generator(featurewise_center = TRUE,
                                  featurewise_std_normalization = TRUE,
                                  rotation_range = 10,
                                  width_shift_range = 0.30,
                                  height_shift_range = 0.30,
                                  horizontal_flip = FALSE  )

#Fit image data generator internal statistics to some sample data
gen_images %>% fit_image_data_generator(X_train)


  model %>% fit_generator(
  flow_images_from_data(X_train, Y_train,gen_images,
                        batch_size=10),
  steps_per_epoch=as.integer(50000/32),epochs = 10,
  validation_data = list(X_test, Y_test) )



#confusion matrix
Confusion=confusionMatrix(predicted_labels, Y_test, positive = NULL)


#history = model %>% fit(X_train, Y_train, epochs = 10, batch_size = 1000,validation_split = 0.2)


