stars = '**************'
print(stars)
print('Importing modules')

import os, operator, itertools, pickle, sys, string, nltk
import numpy as np
import pandas as pd
# pd.options.display.max_columns = 999
pd.options.display.max_rows = 999

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import boxcox
from scipy.special import inv_boxcox

from nltk.corpus import stopwords
from nltk import word_tokenize, FreqDist
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression, SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

np.random.seed(0)

sys.path.insert(0, 'objects')
sys.path.insert(0, 'data')
sys.path.insert(0, 'pickles')

from Axe_Object import Axe

print(stars)
print('Importing JSON Guitars')

file_names = [name for name in os.listdir('data/axe_specs/') if not name.startswith('.')] # Ignores hidden files on mac

axes = []
for filename in file_names:
    try:
        this_axe = Axe('data/axe_listings', 'data/axe_specs', filename)
        if "LOT OF" not in this_axe.title.upper() and this_axe.price > 110 and this_axe.price < 890\
        and "TREMOLO" not in this_axe.title.upper():
            if this_axe.string_config and this_axe.string_config < 5:
                continue
            if this_axe.market != 'EBAY-US':
                continue
            if this_axe.year and this_axe.year > 2019:
                continue
            axes.append(this_axe)
    except ValueError:
        pass

print(f'Getting started training model on {len(axes)} historical auctions')

orig_prices = pd.Series([axe.price for axe in axes], name = 'prices')

bxcx_lam = .3
prices = pd.Series(boxcox([axe.price for axe in axes], lmbda=bxcx_lam), name = 'prices')
plt.figure(figsize = (18,6))
plt.hist(prices, bins=50)
plt.ylabel('Frequency')
plt.xlabel('Guitar Price in USD')
# plt.show()

title_lengths       = pd.Series([axe.len_title for axe in axes], name = 'title_lengths')
auction_duration    = pd.Series([axe.duration for axe in axes], name = 'auction_duration')
shipping_charged    = pd.Series([axe.price_shipping for axe in axes], name = 'shipping_charged')
seller_country      = pd.Series([axe.country_seller for axe in axes], name = 'seller_country')
autopay             = pd.Series([axe.autopay for axe in axes], name = 'autopay')
returns             = pd.Series([axe.returns for axe in axes], name = 'returns')
listing_type        = pd.Series([axe.listing_type for axe in axes], name = 'listing_type')
ship_type           = pd.Series([axe.ship_type for axe in axes], name = 'ship_type')
ship_expedite       = pd.Series([axe.ship_expedite for axe in axes], name = 'ship_expedite')
start_hour          = pd.cut(pd.Series([axe.start_time.hour for axe in axes], name = 'start_hour'), 6)
end_hour            = pd.cut(pd.Series([axe.end_time.hour for axe in axes], name = 'end_hour'), 6)
start_weekday       = pd.Series([axe.start_weekday for axe in axes], name = 'start_weekday').astype('category')
end_weekday         = pd.Series([axe.end_weekday for axe in axes], name = 'end_weekday').astype('category')
returns_time        = pd.Series([axe.returns_time for axe in axes], name = "returns_time").astype('category')
num_pics            = pd.Series([axe.pic_quantity for axe in axes], name = "num_pics")
brand               = pd.Series([axe.brand for axe in axes], name = "brand")
body_type           = pd.Series([axe.body_type for axe in axes], name = "body_type")
color               = pd.Series([axe.color for axe in axes], name = "color")
right_left_handed   = pd.Series([axe.right_left_handed for axe in axes], name = "right_left_handed")
best_offer_enabled  = pd.Series([axe.best_offer_enabled for axe in axes], name = "best_offer_enabled")
country_manufacture = pd.Series([axe.country_manufacture for axe in axes], name = "country_manufacture")

ship_handling_time = pd.Series([axe.ship_handling_time for axe in axes], name = 'ship_handling_time').astype('category')

string_config = pd.cut(pd.Series([axe.string_config for axe in axes], name = "string_config"),
                       [0,5,6,11,20])

seller_feedback_score = pd.cut(pd.Series([axe.seller_feedback_score for axe in axes], name = "seller_feedback_score"), [-411,0,50,100,200,500,750,1250,2500,10000,100000,400000])

seller_positive_percent = pd.cut(pd.Series([axe.seller_positive_percent for axe in axes], name = "seller_positive_percent"), [-10000,99.5,111])

model_year = pd.cut(pd.Series([axe.year for axe in axes], name = "model_year"), [1700,1975,1990,1995,2000,2005,2007,2010,2011,2012,2013,2015])

# ## Text as a Regression Feature
# http://www-stat.wharton.upenn.edu/~stine/research/regressor.pdf
def assemble_guitar_document(axe):
    document = axe.title + ' '
    if axe.year != None:
        document += (str(axe.year) + ' ')
    if axe.material != None:
        document += axe.material + ' '
    if axe.model != None:
        document += axe.model + ' ' 
    if axe.brand != None:
        document += axe.brand + ' '
    if axe.subtitle != None:
        document += axe.subtitle + ' '
    if axe.condition_description != None:
        document += axe.condition_description + ' '
    if axe.description != None:
        document += axe.description
    return document

raw_corpus = [assemble_guitar_document(axe).lower() for axe in axes]

stemmer = SnowballStemmer("english")

stopwords_list = stopwords.words('english') + list(string.punctuation)
stopwords_list += ["''", '""', '...', '``', ",", ".", ":", "'s", "--","’"]

def process_doc(doc):
    stopwords_removed = ''
    tokens = nltk.word_tokenize(doc)
    for i in range(len(tokens)):
        if tokens[i].lower() not in stopwords_list and tokens[i] not in string.punctuation:
            stopwords_removed += stemmer.stem(tokens[i]) + ' '
    return stopwords_removed

print(stars)
print('Processing Text Corpus...')

processed_text = pd.Series(list(map(process_doc, raw_corpus)), name = 'text')


# ## Assemble the Feature Set
y_X_dummies = pd.concat([prices, title_lengths, brand, color, country_manufacture, right_left_handed, best_offer_enabled, shipping_charged, 
               returns, returns_time, autopay, seller_country, ship_handling_time, listing_type, ship_expedite,
               ship_type, num_pics, auction_duration, start_hour, end_hour, start_weekday, end_weekday, 
               seller_positive_percent, model_year, body_type, string_config],
              axis = 1)

y_X = pd.get_dummies(y_X_dummies, drop_first=True)

print(stars)
print('Pickling columns...')

filename = 'pickles/bonus_columns'
outfile = open(filename, 'wb')
pickle.dump(list(y_X.iloc[:,1:].columns), outfile)
outfile.close()

# ### SPLIT

print(stars)
print('Splitting...')

y_X = pd.concat([y_X, processed_text], axis=1)
X_train, X_test, y_train, y_test = train_test_split(y_X.iloc[:,1:], y_X.iloc[:,0], test_size=.20, random_state=42)


print(stars)
print('Scaling...')

# ### Scale It 
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train.iloc[:,:-1]), columns=X_train.iloc[:,:-1].columns)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test.iloc[:,:-1]), columns=X_test.iloc[:,:-1].columns)

outfile = open('pickles/saved_scaler','wb')
pickle.dump(scaler,outfile)
outfile.close()

# Text Features vectorization

vectorizer = TfidfVectorizer(norm=None, ngram_range=(2,3), strip_accents='ascii',
#                             max_df=0.8, min_df=2,
                             max_features=300)

vectorizer.fit(X_train['text'])

print(stars)
print('Vectorizing Text...')

outfile = open('pickles/saved_vectorizer','wb')
pickle.dump(vectorizer,outfile)
outfile.close()

tfidf_train = vectorizer.transform(X_train['text'])
tfidf_train_df = pd.DataFrame(tfidf_train.toarray(), columns=vectorizer.get_feature_names())

tfidf_test = vectorizer.transform(X_test['text'])
tfidf_test_df = pd.DataFrame(tfidf_test.toarray(), columns=vectorizer.get_feature_names())

X_train_ready = pd.concat([X_train.reset_index(drop=True), tfidf_train_df], axis=1).drop('text',axis=1)
X_test_ready = pd.concat([X_test.reset_index(drop=True), tfidf_test_df], axis=1).drop('text',axis=1)

# ### Generate a Tensorboard Projector Visualization:

# import tensorflow as tf
# from tensorflow.contrib.tensorboard.plugins import projector
# board_data = pd.DataFrame(vectorizer.transform(processed_text).toarray(), columns = vectorizer.get_feature_names())
# board_data.shape
# LOG_DIR = 'logs'
# board_data_tf = tf.Variable(board_data, name='board_data_tf')
# metadata = os.path.join(LOG_DIR, 'metadata.tsv')
# with open(metadata, 'w') as metadata_file:
#     for row in [axe.title[:25] for axe in axes]:
#         metadata_file.write('%s\n' % row)
# with tf.Session() as sess:
#     saver = tf.train.Saver([board_data_tf])
#     sess.run(board_data_tf.initializer)
#     saver.save(sess, os.path.join(LOG_DIR, 'board_data_tf.ckpt'))
#     config = projector.ProjectorConfig()
#     # One can add multiple embeddings.
#     embedding = config.embeddings.add()
#     embedding.tensor_name = board_data_tf.name
#     # Link this tensor to its metadata file (e.g. labels).
#     embedding.metadata_path = metadata
#     # Saves a config file that TensorBoard will read during startup.
#     projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

# ### Baseline Naive Error - Guess the Mean Price

price_mean = np.mean(y_train)
price_mean_vector = inv_boxcox([price_mean for i in range(len(y_test))],bxcx_lam)
baseline_error = np.sqrt(mean_squared_error(inv_boxcox(y_test, bxcx_lam), price_mean_vector))
baseline_error

print(stars)
print('Training a Lasso Regressor...')

# ### Lasso Regression
lasso_model = LassoCV(cv=3).fit(X_train_ready, y_train)

outfile = open('pickles/lasso_model','wb')
pickle.dump(lasso_model,outfile)
outfile.close()

y_train_preds = lasso_model.predict(X_train_ready)
y_test_preds = lasso_model.predict(X_test_ready)

y_train_inv = inv_boxcox(y_train, bxcx_lam)
y_test_inv = inv_boxcox(y_test, bxcx_lam)
y_train_preds_inv = inv_boxcox(y_train_preds, bxcx_lam)
y_test_preds_inv = inv_boxcox(y_test_preds, bxcx_lam)

train_error = np.sqrt(mean_squared_error(y_train_inv, y_train_preds_inv))
test_error = np.sqrt(mean_squared_error(y_test_inv, y_test_preds_inv))

print(f'Lasso train_error is {train_error}, lasso test_error is {test_error}\n\n')

print(f'Lasso Train error is a {round((((baseline_error - train_error) / baseline_error) * 100),2)}% improvement over guessing the mean. \n')

print(f'Lasso Test error is a {round((((baseline_error - test_error) / baseline_error) * 100),2)}% improvement over guessing the mean.')

plt.figure(figsize=(8,8))
plt.scatter(y_test_inv, y_test_preds_inv, s=2)
x = np.linspace(100,900, num=2)
plt.plot(x,x)
plt.plot(np.full(len(x),500),x)
plt.plot(x,np.full(len(x),500))
plt.show()

coef = pd.DataFrame(data = lasso_model.coef_, index=X_train_ready.columns)
model_coef = coef.sort_values(by=0).T
model_coef.plot(kind='bar', title='Lasso Coefficients', legend=False, figsize=(16,5))
plt.show()

print(stars)
print('Writing out Lasso Coefs...')

t_mod = model_coef.T
lasso_feats = t_mod[abs(t_mod) > 0].dropna()
lasso_feats.to_csv('lasso_coefs.csv')
# Can I get a confusion matrix?

price_thresh = 510

high_actual = y_test_inv > price_thresh

high_preds = y_test_preds_inv > price_thresh
plt.figure(figsize=(5,5))
cnf_matrix = confusion_matrix(high_actual, high_preds)
plt.imshow(cnf_matrix,  cmap=plt.cm.Blues) #Create the basic matrix.

#Add title and Axis Labels
plt.title('Lasso Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')

#Add appropriate Axis Scales
class_names = ['Under', 'Over'] #Get class labels to add to matrix
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

#Add Labels to Each Cell
thresh = cnf_matrix.max() / 2. #Used for text coloring below
#Here we iterate through the confusion matrix and append labels to our visualization.
for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

plt.show()


bad_guesses = np.full(len(y_test_inv), True)
bad_cnf_matrix = confusion_matrix(high_actual, bad_guesses)
print(f'If you just guessed every guitar would sell for more than ${price_thresh} you\'d only be right {round((bad_cnf_matrix[1][1] / (bad_cnf_matrix[1][1] + bad_cnf_matrix[0][1]))*100, 2)}% of the time.')

print(f'Lasso model only identifies {round(100 * cnf_matrix[1][1] / (cnf_matrix[1][0] + cnf_matrix[1][1]), 2)}% of the guitars that will sell above ${price_thresh}.')

print(f'However, when lasso guesses above ${price_thresh}, it\'s correct {round((cnf_matrix[1][1] / (cnf_matrix[1][1] + cnf_matrix[0][1]))*100, 2)}% of the time.')


# ## Fitting a TPOT Auto-ML Pipeline
# from tpot import TPOTRegressor
# pipeline_optimizer = TPOTRegressor(generations=80, population_size=100,
#                          offspring_size=None, mutation_rate=0.9,
#                          crossover_rate=0.1,
#                          scoring='neg_mean_squared_error', cv=5,
#                          subsample=1.0, n_jobs=8,
#                          max_time_mins=570, max_eval_time_mins=5,
#                          random_state=None, config_dict='TPOT sparse',
#                          warm_start=False,
#                          memory=None,
#                          use_dask=False,
#                          periodic_checkpoint_folder='TPOT_saves',
#                          early_stop=None,
#                          verbosity=3,
#                          disable_update_check=False)

# X_train_ready_tpot = X_train_ready.values.astype('float')
# pipeline_optimizer.fit(X_train_ready_tpot, y_train)

# Export:

# pipeline_optimizer.export('pickles/tpot_guitar_pipeline.py')

# ### Import TPOT-Selected Model
# (Copy paste ideal pipeline from tpot_guitar_pipeline.py file)

tpot = make_pipeline(SelectPercentile(score_func=f_regression, percentile=85), 
                                  StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, 
                                                                           max_depth=5, 
                                                                           min_child_weight=16, 
                                                                           n_estimators=100, nthread=1, 
                                                                           subsample=0.6500000000000001)),
    
                                  RandomForestRegressor(bootstrap=False, 
                                                        max_features=0.1, 
                                                        min_samples_leaf=3, 
                                                        min_samples_split=9, 
                                                        n_estimators=100)
                                )

print(stars)
print('Training TPOT Pipeline...')

tpot.fit(X_train_ready.values.astype('float'), y_train)

print(stars)
print('Pickling the TPOT...')

filename = 'pickles/tpot'
outfile = open(filename, 'wb')
pickle.dump(tpot, outfile)
outfile.close()

y_train_preds_tpot = tpot.predict(X_train_ready.values.astype('float'))
y_test_preds_tpot = tpot.predict(X_test_ready.values.astype('float'))

y_train_preds_inv_tpot = inv_boxcox(y_train_preds_tpot, bxcx_lam)
y_test_preds_inv_tpot = inv_boxcox(y_test_preds_tpot, bxcx_lam)

train_error_tpot = np.sqrt(mean_squared_error(y_train_inv, y_train_preds_inv_tpot))
test_error_tpot = np.sqrt(mean_squared_error(y_test_inv, y_test_preds_inv_tpot))

print(f'TPOT Train RMSE: {round(train_error_tpot, 2)} --- Test RMSE: {round(test_error_tpot, 2)}\n')

print(f'TPOT Train error is a {round((((baseline_error - train_error_tpot) / baseline_error) * 100),2)}% improvement over guessing the mean. \n')

print(f'TPOT Test error is a {round((((baseline_error - test_error_tpot) / baseline_error) * 100),2)}% improvement over guessing the mean.')

plt.figure(figsize=(8,8))
plt.scatter(y_test_inv, y_test_preds_inv_tpot, s=2)
x = np.linspace(100,900, num=2)
plt.plot(x,x)
plt.plot(np.full(len(x),500),x)
plt.plot(x,np.full(len(x),500))
plt.show()


price_thresh = 600

high_preds_tpot = y_test_preds_inv_tpot > price_thresh
plt.figure(figsize=(5,5))
cnf_matrix_tpot = confusion_matrix(high_actual, high_preds_tpot)
plt.imshow(cnf_matrix_tpot,  cmap=plt.cm.Blues) #Create the basic matrix.

#Add title and Axis Labels
plt.title('TPOT Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')

#Add appropriate Axis Scales
class_names = ['Under', 'Over'] #Get class labels to add to matrix
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

#Add Labels to Each Cell
thresh = cnf_matrix_tpot.max() / 2. #Used for text coloring below
#Here we iterate through the confusion matrix and append labels to our visualization.
for i, j in itertools.product(range(cnf_matrix_tpot.shape[0]), range(cnf_matrix_tpot.shape[1])):
        plt.text(j, i, cnf_matrix_tpot[i, j],
                 horizontalalignment="center",
                 color="white" if cnf_matrix_tpot[i, j] > thresh else "black")

plt.show()

print(f'TPOT model only identifies {round(100 * cnf_matrix_tpot[1][1] / (cnf_matrix_tpot[1][0] + cnf_matrix_tpot[1][1]), 2)}% of the guitars that will sell above ${price_thresh}.')

print(f'However, when TPOT guesses above ${price_thresh}, it\'s correct {round((cnf_matrix_tpot[1][1] / (cnf_matrix_tpot[1][1] + cnf_matrix_tpot[0][1]))*100, 2)}% of the time.')


# ****

# V2:
# * How much white balance a guitar's pictures need, as a feature?
# * Filter by number of unique bidders