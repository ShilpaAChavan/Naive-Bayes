################################# Naive Bayes ########################################

#Objective: Build a naive Bayes model on the data set for classifying the ham and spam

#Data : sms_raw_NB.csv
######################################################################################
install.packages("tm")
library(tm)

##Step : Data Exploration 
# read the sms data into the sms data frame
sms_raw <- read.csv('D:\\Shilpa\\Datascience\\Assignments\\Naive Bayes\\sms_raw_NB.csv', stringsAsFactors = FALSE)

# examine the structure of the sms data
str(sms_raw)
#'data.frame':	5559 obs. of  2 variables:
#  $ type: chr  "ham" "ham" "ham" "spam" ...
#$ text: chr  "Hope you are having a good week. Just checking in" "K..give back my thanks." 
#"Am also doing in cbe only. But have to pay." "complimentary 4 STAR Ibiza Holiday or å£10,000 
#cash needs your URGENT collection. 09066364349 NOW from Landline"| __truncated__ ..

# convert spam/ham to factor.
sms_raw$type <- factor(sms_raw$type)
table(sms_raw$type)
#ham spam 
#4812  747 

summary(sms_raw)
#type               text          
#Length:5559        Length:5559       
#Class :character   Class :character  
#Mode  :character   Mode  :character 

#Data includes total 5559 messages,4812 ham messages and 747 spam messages.

# Step: cleaning the data
# build a corpus using the text mining (tm) package
sms_corpus <- Corpus(VectorSource(sms_raw$text))

# clean up the corpus using tm_map()
# Convert to lower case
corpus_clean <- tm_map(sms_corpus, tolower)
# remove numbers
corpus_clean <- tm_map(corpus_clean, removeNumbers)
# remove stopwords e.g. to, and, but,the etc
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
# remove punctuation
corpus_clean <- tm_map(corpus_clean, removePunctuation)
# remove extra whitespaces
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

inspect(corpus_clean)
#Inspect the same 4 documents to visualize
corpus_clean[[1]]$content
#[1] "hope good week just checking "

corpus_clean[[2]]$content
#[1] "kgive back thanks"

corpus_clean[[400]]$content
#[1] "urgent ur awarded complimentary
#trip eurodisinc trav acoentry å£ claim txt dis å£morefrmmob shracomorsglsuplt ls aj"

corpus_clean[[1000]]$content
#[1] "kk tomorrow onwards started ah"

##Step: Feature Engineering
# create a document-term sparse matrix
sms_dtm <- DocumentTermMatrix(corpus_clean, control = list(global = c(2, Inf)))
print(sms_dtm)
#<<DocumentTermMatrix (documents: 5559, terms: 7925)>>
#  Non-/sparse entries: 42663/44012412
#Sparsity           : 100%

# creating training and test datasets
sms_raw_train <- sms_raw[1:4169, ]
sms_raw_test  <- sms_raw[4170:5559, ]

sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test  <- sms_dtm[4170:5559, ]

sms_corpus_train <- corpus_clean[1:4169]
sms_corpus_test  <- corpus_clean[4170:5559]


# check that the proportion of spam is similar
prop.table(table(sms_raw_train$type))
#ham      spam 
#0.8647158 0.1352842 

prop.table(table(sms_raw_test$type))
#ham      spam 
#0.8683453 0.1316547 

# Word cloud visualization
library(wordcloud)
wordcloud(sms_corpus_train, min.freq = 30, random.order = FALSE)

# subset the training data into spam and ham groups
spam <- subset(sms_raw_train, type == "spam")
ham  <- subset(sms_raw_train, type == "ham")
wordcloud(spam$text, max.words = 40, scale = c(3, 0.5),colors = 'blue')
wordcloud(ham$text, max.words = 40, scale = c(4, 0.5),colors = 'blue')

# indicator features for frequent words
sms_dict<-findFreqTerms(sms_dtm_train, 3) #find words that appears at least 3 times
summary(sms_dict)
#   Length     Class      Mode 
#1953 character character 
head(sms_dict)
# [1] "checking" "good"     "hope"     "just"     "week"     "back" 
sms_train <- DocumentTermMatrix(sms_corpus_train, list(dictionary = sms_dict))
sms_test  <- DocumentTermMatrix(sms_corpus_test, list(dictionary = sms_dict))


# convert counts to a factor
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}

# apply() convert_counts() to columns of train/test data
sms_train <- apply(sms_train, MARGIN = 2, convert_counts)
sms_test  <- apply(sms_test, MARGIN = 2, convert_counts)
#sms_train
#sms_test


## Step: Training a model on the data ----
library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_raw_train$type)
summary(sms_classifier)
#Length Class  Mode     
#apriori      2   table  numeric  
#tables    1953   -none- list     
#levels       2   -none- character
#isnumeric 1953   -none- logical  
#call         3   -none- call    

## Step: Evaluating model performance ----
sms_test_pred <- predict(sms_classifier, sms_test)
library(gmodels)
CrossTable(sms_test_pred, sms_raw_test$type,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

Accuracy(sms_test_pred, sms_raw_test$type)
#0.9769784
#Cell Contents
#  |-------------------------|
#  |                       N |
#  |           N / Col Total |
#  |-------------------------|
  
#  Total Observations in Table:  1390 

#               |            actual 
#predicted      |       ham |      spam | Row Total | 
#  -------------|-----------|-----------|-----------|
#  ham          |      1203 |        28 |      1231 | 
#               |     0.997 |     0.153 |           | 
#  -------------|-----------|-----------|-----------|
#  spam         |         4 |       155 |       159 | 
#               |     0.003 |     0.847 |           | 
#  -------------|-----------|-----------|-----------|
#  Column Total |      1207 |       183 |      1390 | 
#               |     0.868 |     0.132 |           | 
#  -------------|-----------|-----------|-----------|
  
## Step: Improving model performance ----
sms_classifier2 <- naiveBayes(sms_train, sms_raw_train$type, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
CrossTable(sms_test_pred2, sms_raw_test$type,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
#Cell Contents
#  |-------------------------|
#  |                       N |
#  |           N / Col Total |
#  |-------------------------|
  
#  Total Observations in Table:  1390 

#               | actual 
#    predicted  |       ham |      spam | Row Total | 
#  -------------|-----------|-----------|-----------|
#  ham          |      1205 |        30 |      1235 | 
#               |     0.998 |     0.164 |           | 
#  -------------|-----------|-----------|-----------|
#  spam         |         2 |       153 |       155 | 
#               |     0.002 |     0.836 |           | 
#  -------------|-----------|-----------|-----------|
#  Column Total |      1207 |       183 |      1390 | 
#               |     0.868 |     0.132 |           | 
#  -------------|-----------|-----------|-----------|

install.packages("MLmetrics")
require("MLmetrics")
Accuracy(sms_test_pred2, sms_raw_test$type)
#0.9769784
  
#Looking at the table we can see that 32 messages out of 1390 messages
#have been incorrectly classified as spam or ham. The model has an accuracy of 0.9769784 



