rm(list=ls())
setwd("C:/Users/hp/Desktop/minor-harsh/images")
library("keras")
dataset = read.csv("Imagedataset.csv")

library(caTools)
set.seed(123)

split = sample.split(dataset$labels, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

X_train = as.matrix(training_set[-4097])
Y_train = unlist(training_set[4097])

X_test = as.matrix(test_set[-4097])
Y_test = unlist(test_set[4097])

dim(X_test)=c(nrow(X_test),64,64,1)
dim(X_train)=c(nrow(X_train),64,64,1)


Y_train = to_categorical(Y_train, num_classes = 15)
Y_test = to_categorical(Y_test, num_classes = 15)


model <- keras_model_sequential()

model%>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),activation = 'relu', 
                input_shape = c(64,64,1)) %>%
  layer_max_pooling_2d(pool_size = c(3,3)) %>%
  
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu')  %>%
  layer_max_pooling_2d(pool_size = c(3,3)) %>%
  layer_dropout(rate = 0.15) %>%
  
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu')  %>%
  layer_max_pooling_2d(pool_size = c(3,3)) %>%
  
  
  layer_flatten() %>% 
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 15, activation = 'softmax')%>%
  
  compile(
    loss = 'categorical_crossentropy', 
    optimizer = optimizer_adam(),
    metrics=c("accuracy"))

gen_images = image_data_generator(featurewise_center = TRUE,
                                   featurewise_std_normalization = TRUE,
                                   rotation_range = 20,
                                   width_shift_range = 0.30,
                                   height_shift_range = 0.30,
                                   horizontal_flip = TRUE  )
#Fit image data generator internal statistics to some sample data
gen_images %>% fit_image_data_generator(X_train)


model %>% fit_generator(
  flow_images_from_data(X_train, Y_train,gen_images,
                        batch_size=32,save_to_dir="C:/Users/hp/Desktop/minor-harsh/augmentedimages"),
  steps_per_epoch=as.integer(50000/32),epochs = 10,
  validation_data = list(X_test, Y_test) )

# predicted labels
predicted_labels = predict(model, newdata = X_test, type = "response")

#confusion matrix
Confusion=confusionMatrix(predicted_labels, Y_test, positive = NULL)

# to save model 
#saveRDS(titanic_glm, file = "model.Rds", compress = TRUE)


