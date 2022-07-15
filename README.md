# Passion_projects
Implemented 2 logistic regression machine learning algorithms. The first is used in order to assess whether or not
a user would have watched a netflix film given previous filmed that they have already watched, which utilizes a test
set of 19 different movies that many users watched and uses that to and whether they have watched X movie. Then, it
uses this data to make future predictions on whether someone is likely to watch X movie. The second algorithm looks
at user cell phone reviews, and uses the presence of negative words in order to classify as review as positive or negative.
It then uses the trained data to make assessments on whether a review will be positive or negative using the words used in the review.
The Netflix Algorithm is 66% accurate, and the cell-phone review algorithm is 74% accurate.

The phone reviews algorithm goes through 65,000 reviews in order to assess whether they are positive or negative based on the words they use, derived
from the top positive and negative words found in reviews, it uses this to convert reviews into binary data of a "positive" review and a "negative" review.
It then uses the ML algorithm to classify these reviews as positive or negative, and compares its accuracy to the actual rating of the review, where anything
BELOW a score of 3/5 is deemed a negative review.

The Netflix algorithm sifts through 19 movies, and then sees based on a user's watch history for those 19 movies, whether they have watched X movie.
It looks at over 100+ users in order to train the data, and then tests the data on 500+ users in order to determine whether a user has watched X movie
given their watch history of the 19 movies selected.

Positive & Negative Words Found Through:

Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews." 
;       Proceedings of the ACM SIGKDD International Conference on Knowledge 
;       Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle, 
;       Washington, USA, 
Bing Liu, Minqing Hu and Junsheng Cheng. "Opinion Observer: Analyzing 
;       and Comparing Opinions on the Web." Proceedings of the 14th 
;       International World Wide Web conference (WWW-2005), May 10-14, 
;       2005, Chiba, Japan.

The phone training data is linked in githib, however, the test data had 1.4 million reviews and was thus too big to fit in github. It is linked below:
https://www.kaggle.com/datasets/masaladata/14-million-cell-phone-reviews?select=phone_user_review_file_2.csv

The training data, though linked in github is also found here:
https://www.kaggle.com/datasets/grikomsn/amazon-cell-phones-reviews?select=20191226-reviews.csv

log_regress.py is the Netflix Project

phone_review_analysis is the Phone Review Project

Both of these projects use a logistic regression machine learning model. It is VERY important to note that despite the model's accuracy machine learning
is constantly developing, and by only looking at specific points of data such as reviews, or past watched movies and neglecting other factors such as,
region, demographics, etc, these models are meant only to be used as predictors, not as fact. There is a lot of data that is not investigated and left behind
thus, it is important to note the drawbacks of only using particular points of data to make a prediction. This is a predictive model and should not be used
to draw any conclusions without further research and evidence.
