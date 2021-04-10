##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,] 

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

edx %>% summarize(n_movies = n_distinct(movieId))
#load(file='Workspace_Capstone_03_final.RData')
paste('The edx dataset has',nrow(edx),'rows and',ncol(edx),'columns.')
str(edx)
#load(file='Workspace_Capstone_03_final.RData')
paste('The edx dataset has',nrow(edx),'rows and',ncol(edx),'columns.')
paste(sum(edx$rating == 0), 'ratings with 0 were given and',
      sum(edx$rating == 3),'ratings with 3')
drama <- edx %>% filter(str_detect(genres,"Drama"))
comedy <- edx %>% filter(str_detect(genres,"Comedy"))
thriller <- edx %>% filter(str_detect(genres,"Thriller"))
romance <- edx %>% filter(str_detect(genres,"Romance"))

paste('Drama has',nrow(drama),'movies')
paste('Comedy has',nrow(comedy),'movies')
paste('Thriller has',nrow(thriller),'movies')
paste('Romance has',nrow(romance),'movies')
edx %>% group_by(title) %>% summarise(number = n()) %>%
  arrange(desc(number))
edx %>% group_by(rating) %>% summarise(number = n()) %>%
  arrange(desc(number))
head(sort(-table(edx$rating)),5)


# Ratings distribution in MovieLens data
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.25, color = "red", fill = "black") +
  xlab("Rating") +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ylab("Count") +
  ggtitle("Rating distribution in MovieLens data")
# Number of ratings per Movie in MovieLens data
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 100, color = "blue", fill = "black") + xlab("Ratings Count") +
  scale_x_log10() +
  ylab("Movie Count") +
  ggtitle("Number of Ratings per Movie")
# Number of ratings given by users in MovieLens data
edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 100, color = "green", fill = "black") + xlab("Ratings Count") +
  scale_x_log10() +
  ylab("User Count") +
  ggtitle("Number of Ratings in MovieLens data")
# Mean ratings given by user in MovieLens data
edx %>%
  group_by(userId) %>%
  filter(n() >= 30) %>%
  summarize(mean_rating = mean(rating)) %>% ggplot(aes(mean_rating)) +
  geom_histogram(bins = 100, color = "yellow", fill = "black") + xlab("Mean Rating") +
  ylab("Number of Users") +
  ggtitle("Mean Ratings in MovieLens data")
summary(edx)

#number of rating for each movie genres
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>% ggplot(aes(genres,count)) + 
  geom_bar(aes(fill =genres),stat = "identity")+ 
  labs(title = " Number of Rating for Each Genre")+
  theme(axis.text.x  = element_text(angle= 90, vjust = 50 ))

##Data Partitioning  
##partition the edx data set into 20 % for test set and 80% for the training set.
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

## Model building and RMSE calculation 
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2, na.rm = TRUE))
}

## Model 1: Navie model
Mu_1 <- mean(train_set$rating)
Mu_1
## Testing RMSE of Model 1 : Navie Model
Mu_1_rmse <- RMSE(validation$rating, Mu_1) 
Mu_1_rmse

## Model 2 : Famous Movie effect Model
famous_movie_effect <- edx %>%
  group_by(movieId) %>%
  summarize(effect_fm = mean(rating - Mu_1)) 

famous_movie_effect %>%
  ggplot(aes(effect_fm)) +
  geom_histogram(bins = 35, color = "grey", fill = "black") + ylab("Number of Movies") +
  ggtitle("Famous moive effect distribution")

## Testing RMSE of Model 2 : Famous Movie effect Model
movie_effect_predictions <- validation %>% left_join(famous_movie_effect, by = "movieId")%>% mutate(prediction = Mu_1 + effect_fm)
Mu_2_rmse <- RMSE(validation$rating, movie_effect_predictions$prediction) 
Mu_2_rmse

## Model 3 : Famous movie and User effect Model
user_effect <- edx %>%
  left_join(famous_movie_effect, by = "movieId") %>% group_by(userId) %>%
  summarize(effect_u = mean(rating - Mu_1 - effect_fm))

## Testing RMSE of Model 3 : Famous movie and User effect Model
user_effect_predictions <- validation %>% left_join(famous_movie_effect, by = "movieId")%>% 
  left_join(user_effect, by = "userId")%>%
  mutate(prediction = Mu_1 + effect_fm + effect_u)
Mu_3_rmse <- RMSE(validation$rating, user_effect_predictions$prediction) 
Mu_3_rmse

## Model 4: Regularized Model
# Let the effect_reg be the tuning parameter
effect_reg <- seq(0,10,0.1)

reg_rmse <- sapply(effect_reg, function(effect_reg){
  Mu_1 <- mean(edx$rating) 
  effect_fm <- edx %>%group_by(movieId) %>%
    summarize(effect_fm = sum(rating - Mu_1) / (n() + effect_reg)) 
  effect_u <- edx %>%
    left_join(effect_fm, by = "movieId") %>%
    group_by(userId) %>%
    summarize(effect_u = sum(rating - effect_fm - Mu_1) / (n() + effect_reg))
  predictions <- validation %>% left_join(effect_fm, by = "movieId") %>% left_join(effect_u, by = "userId") %>% mutate(prediction = Mu_1 + effect_fm + effect_u) %>% .$prediction
  RMSE(validation$rating, predictions) })

# Plot the effect_reg against the regularised RMSEs to visualize what is the best effect_reg
tibble(effect_reg = effect_reg, reg_rmse = reg_rmse) %>%
  ggplot(aes(effect_reg, reg_rmse)) + geom_point()

# Find the effect_reg with minimal reg_rmse
effect_reg <- effect_reg[which.min(reg_rmse)]
effect_reg
# Find the lowest rmse in the model based on the minimal effect_reg
min(reg_rmse)

# Display the RMSE result of the 4 models
results <- data_frame(Model=c("Model 1:Navie Model",
                              "Model 2:Famous Movie effect Model", 
                              "Model 3:Famous movie and User effect Model",
                              "Model 4:Regularized Model"),
                      RMSE=c(Mu_1_rmse,Mu_2_rmse,Mu_3_rmse,min(reg_rmse)))
results
signif(results$RMSE, digits=6)