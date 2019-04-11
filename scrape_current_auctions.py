from key import API_KEY

stars = '**************'
print(stars)
print('Importing modules')

import requests, json, os, pickle, yagmail, sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.max_columns = 999
pd.options.display.max_rows = 999

from scipy.special import inv_boxcox

import nltk, string, os
from nltk.corpus import stopwords
from nltk import word_tokenize, FreqDist
from nltk.stem.snowball import SnowballStemmer

from scipy.stats import boxcox

from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, 'objects')
from Shade_Sale_Memory import Shade_Sale_Memory
sys.path.insert(0, 'pickles')

# Trawl for Prospective Deals
hours_ahead = 6 #int(input("Enter how many hours ahead you want to scrape for: "))

API_KEY = API_KEY # Enter your API Key/"App ID" Here. Mine was 40 chars long.

FIND_ADVANCED = "findItemsAdvanced" # This is the eBay API endpoint service we'll be querying.
# ELEC_GUITARS = '33034'
MENS_SUNGLASSES = '79720'
USED = '3000'
AUCTION = "Auction"
AUCTIONWITHBIN = "AuctionWithBIN"

ct = datetime.utcnow()
now = datetime.now()

endtime_datetime = now + timedelta(hours=hours_ahead)

end_time = f'{ct.year}-'

if len(str(ct.month)) < 2:
    end_time += '0'
end_time += f'{ct.month}-'

day_incrementer = (ct.hour + hours_ahead) // 24

if int(ct.day + day_incrementer) < 10:
    end_time += '0'
end_time += str(ct.day + day_incrementer) + 'T'

if len(str((ct.hour + hours_ahead)%24)) < 2:
    end_time += '0'
end_time += f'{(ct.hour + hours_ahead) % 24}:'

if len(str(ct.minute)) < 2:
    end_time += '0'
end_time += f'{ct.minute}:'

if len(str(ct.second)) < 2:
    end_time += '0'
end_time += f'{ct.second}.'

if len(str(ct.microsecond)) < 2:    
    end_time += '0'
end_time += str(ct.microsecond)[:3] + 'Z'

ITEM_FILTER_0 = f'itemFilter(0).name=Condition&itemFilter(0).value={USED}' # Only used items
ITEM_FILTER_1 = f'itemFilter(1).name=HideDuplicateItems&itemFilter(1).value=true' # No duplicate listings
ITEM_FILTER_2 = f'itemFilter(2).name=MinPrice&itemFilter(2).value=1' # Only items that sell for > this value
ITEM_FILTER_3 = f'itemFilter(3).name=MaxQuantity&itemFilter(3).value=1' # No lots or batch sales. One item at a time
ITEM_FILTER_4 = f'itemFilter(4).name=MaxPrice&itemFilter(4).value=200' # Only items going for < this value
ITEM_FILTER_5 = f'itemFilter(5).name=EndTimeTo&itemFilter(5).value={end_time}' # Only ending soonish

def find_current_auctions(PAGE):#, keywords):
    '''Make a request to the eBay API and return the JSON text of this page number'''
    r = requests.get(
                 f'https://svcs.ebay.com/services/search/FindingService/v1?'
                 f'OPERATION-NAME={FIND_ADVANCED}&'
                 f'X-EBAY-SOA-SECURITY-APPNAME={API_KEY}&'
                 f'RESPONSE-DATA-FORMAT=JSON&'
                 f'REST-PAYLOAD&'
                 f'categoryId={MENS_SUNGLASSES}&'
                 f'descriptionSearch=true&'
                 f'{ITEM_FILTER_0}&' # USED
                 f'{ITEM_FILTER_1}&' # NO DUPES
                 f'{ITEM_FILTER_2}&' # MINPRICE
                 f'{ITEM_FILTER_3}&' # NO LOTS
                 f'{ITEM_FILTER_4}&' # MAX PRICE
                 f'{ITEM_FILTER_5}&' # END TIME
#                  f'keywords={keywords}&'
                 f'paginationInput.pageNumber={str(PAGE)}') # value to be looped through when collecting lotsa data
    if r.json()['findItemsAdvancedResponse'][0].get('searchResult'):
        return r.json()['findItemsAdvancedResponse'][0]['searchResult'][0]['item']
    else:
        return None

def get_specs(ITEM_ID):
    '''Return the specifics of a single eBay auction. String input.'''
    res = requests.get('http://open.api.ebay.com/shopping?'
                    f'callname=GetSingleItem&'
                    f'responseencoding=JSON&'
                    f'appid={API_KEY}&'
                    f'version=967&' # What is this?
                    f'ItemID={ITEM_ID}&'
                    f'IncludeSelector=Details,ItemSpecifics,TextDescription')
    try:
        return res.json()['Item']
    except KeyError:
        pass

def trawl_for_items(start_page, stop_page, fetch_function):#, keywords):
    '''Spams the eBay API for pages of data'''
    j = 0
    k = 0
    existing_item_ids = []
    listings = []
    
    for i in range(start_page+1, stop_page+1):
        page = fetch_function(i)#, keywords)
        if page:
            for item in page:
                k += 1
                if item['itemId'][0] not in existing_item_ids:
                    existing_item_ids.append(item['itemId'][0])
                    j += 1
                    print(f'Get {j}')
                    listings.append({'listing': item,
                                     'specs': get_specs(item['itemId'][0])})
                else:
                    print('Skip')
    
    print(f'\nChecked {k} items')
    print(f'\nGot {j} new items')
    
    return listings

prospects = []

print(stars)
print('Trawling for goodies on the online:')

# See this for eBay keyword formatting: https://developer.ebay.com/Devzone/finding/Concepts/FindingAPIGuide.html#usekeywords
# URL Formatting can be found here: https://www.freeformatter.com/url-encoder.html
prospects.extend(trawl_for_items(0,3,find_current_auctions))#,'american+-%28squier%2Csquire%2Cepiphone%2Cepi%29'))

print(stars)
print('De-Serializing what we got from eBay:')

# specs = db.specs
items = []
for prospect in prospects:
    if prospect.get('specs'):
        try:
            this_item = Shade_Sale_Memory(prospect['listing'],prospect['specs'])
            if "BAG" not in this_item.title.upper():
                if "LOT OF" not in this_item.title.upper():
                    if this_item.end_time > endtime_datetime:
                        print('auction not ending soon enough')
                        continue
                    if this_item.end_time < now:
                        print('this auction ended')
                        continue
                    else: items.append(this_item)
                else:
                    print('this is a lot'); continue
            else: 
                print('this is a bag'); continue
        except ValueError:
            print("val_error")
            pass
    else:
        print("no specs")
        pass

print(stars)
print('More organizing data:')

# Properties
title_lengths       = pd.Series([item.len_title for item in items], name = 'title_lengths')
auction_duration    = pd.Series(np.full(len(items),7*24), name = 'auction_duration')
shipping_charged    = pd.Series(np.full(len(items),0), name = 'shipping_charged')          
seller_country_US   = pd.Series([1 for i in range(len(items))], name = 'seller_country_US')
autopay             = pd.Series([False for item in range(len(items))], name = 'autopay')
returns             = pd.Series([False for item in range(len(items))], name = 'returns')
listing_type_FixedPrice = pd.Series([True for item in range(len(items))], name = 'listing_type_FixedPrice')
ship_type_Free      = pd.Series([1 for item in range(len(items))], name = 'ship_type_Free')
ship_expedite       = pd.Series([0 for item in range(len(items))], name = 'ship_expedite')
start_hour          = pd.Series([1 for item in range(len(items))], name = 'start_hour_(7.667, 11.5]')
end_hour            = pd.Series([1 for item in range(len(items))], name = 'end_hour_(7.667, 11.5]')
start_weekday_6     = pd.Series([6 for item in range(len(items))], name = 'start_weekday_6')
end_weekday_6       = pd.Series([6 for item in range(len(items))], name = 'end_weekday_6')
brand               = pd.Series([item.brand for item in items], name = "brand")
returns_time        = pd.Series([0 for item in range(len(items))], name = 'returns_time')
num_pics            = pd.Series([12 for item in range(len(items))], name = 'num_pics')
best_offer_enabled  = pd.Series([True for item in range(len(items))], name = 'best_offer_enabled')
ship_handling_time_2= pd.Series([1 for item in range(len(items))], name = 'ship_handling_time_2')
seller_positive_percent = pd.Series([1 for item in range(len(items))], name = 'seller_positive_percent_(99.5, 111.0]')
frame_color         = pd.Series([item.frame_color for item in items], name = "frame_color")
lens_color          = pd.Series([item.lens_color for item in items], name = "lens_color")
frame_material      = pd.Series([item.frame_material for item in items], name = "frame_material")
lens_tech           = pd.Series([item.lens_tech for item in items], name = "lens_tech")
country_manufacture = pd.Series([item.country_manufacture for item in items], name = "country_manufacture")
temple_length_listed= pd.Series([item.temple_length_binary for item in items], name = "temple_length_listed")
style               = pd.Series([item.style for item in items], name = "style")
protection          = pd.Series([item.protection for item in items], name = "protection")

feedback_lmbda = -.02
seller_feedback_score_boxed = pd.Series(boxcox([item.seller_feedback_score + 5 for item in items], lmbda=feedback_lmbda), name='seller_feedback_score_boxed')


X_dummies = pd.concat([title_lengths, 
                       brand, 
                       frame_color, 
                       frame_material, 
                       lens_color,
                       lens_tech,
                       country_manufacture, 
                       best_offer_enabled, 
                       shipping_charged, 
                       returns,
                       returns_time,
                       autopay, 
                       ship_handling_time_2, 
                       listing_type_FixedPrice, 
                       ship_expedite,
                       ship_type_Free, 
                       num_pics, 
                       auction_duration, 
                       start_hour, 
                       end_hour, 
                       start_weekday_6, 
                       end_weekday_6, 
                       seller_feedback_score_boxed,
                       style,
                       protection,
                       temple_length_listed],
              axis = 1)

X_nontext = pd.get_dummies(X_dummies, drop_first=True)

print(stars)
print('Text Processing:')

# ## Text as a Regression Feature
# http://www-stat.wharton.upenn.edu/~stine/research/regressor.pdf
def assemble_guitar_document(item):
    document = item.title + ' '
    if item.frame_color != 'UNLISTED':
        document += item.frame_color + ' '
    if item.lens_color != 'UNLISTED':
        document += item.lens_color + ' '
    if item.frame_material != 'UNLISTED':
        document += item.frame_material + ' '
    if item.model != 'UNLISTED':
        document += item.model + ' ' 
    if item.style != 'UNLISTED':
        document += item.style + ' '
    if item.brand != 'UNLISTED':
        document += item.brand + ' '
    if item.lens_tech != 'UNLISTED':
        document += item.lens_tech + ' '
    if item.protection != 'UNLISTED':
        document += item.protection + ' '
    if item.subtitle != None:
        document += item.subtitle + ' '
    if item.condition_description != None:
        document += item.condition_description + ' '
    if item.description != None:
        document += item.description
    return document

raw_corpus = [assemble_guitar_document(item).lower() for item in items]

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

processed_text = pd.Series(list(map(process_doc, raw_corpus)), 
                           name = 'text')


print(stars)
print('Importing saved vectorizer:')


infile = open('pickles/saved_vectorizer','rb')
vectorizer = pickle.load(infile)
infile.close()

print(stars)
print('TF-IDF Transform:')

tfidf = vectorizer.transform(processed_text)
tfidf_df = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())

X_prune = pd.concat([X_nontext, tfidf_df], axis=1)

infile = open('pickles/bonus_columns','rb')
bonus_columns = pickle.load(infile)
infile.close()


fillers = []
for col in bonus_columns:
    if col not in X_prune.columns:
        filler = pd.Series(np.full(len(items),0), name=col)
        fillers.append(filler)
for col in X_prune.columns:
    if col not in bonus_columns:
        X_prune.drop(col, axis=1, inplace=True)
fillers_df = pd.concat(fillers, axis=1)

X = pd.concat([X_prune, fillers_df], axis=1)


print(stars)
print('Fetching the fitted scaler:')

infile = open('pickles/saved_scaler','rb')
scaler = pickle.load(infile)
infile.close()

X_ready_scaled = pd.DataFrame(scaler.transform(X))


print(stars)
print('Import Trained Regressor:')


infile = open('pickles/tpot','rb')
tpot = pickle.load(infile)
infile.close()


print(stars)
print('Generate estimates:')

y_preds = tpot.predict(X_ready_scaled)

bxcx_lam = 0
y_preds_inv = inv_boxcox(y_preds, bxcx_lam)


bids = []
for item in items:
    if item._Shade_Sale_Memory__body['listing']['sellingStatus'][0].get('bidCount'):
        bids.append(item._Shade_Sale_Memory__body['listing']['sellingStatus'][0]['bidCount'][0])
    else:
        bids.append('0')

predicted_df = pd.concat([pd.Series(y_preds_inv), pd.Series([item.initial_price + item.price_shipping for item in items]),
                          pd.Series(y_preds_inv) / pd.Series([item.initial_price for item in items]),
                          pd.Series([item.title for item in items]), 
                          pd.Series([item.url for item in items]), 
                         pd.Series([item.pic for item in items]),
                         pd.Series(bids),
                         pd.Series([round((item.end_time - ct).seconds/60/60, 2) for item in items])],
                         axis=1)

predicted_df.columns = ['Estimate', 'Price', 'Ratio','Title','Link', 'Pic', 'Bids', 'Hours_til_close']

highest_value = predicted_df[predicted_df.Price < predicted_df.Estimate].sort_values('Estimate', ascending=False)
hv_10 = highest_value.iloc[:10,:]

most_underrated = predicted_df[predicted_df.Price < predicted_df.Estimate].sort_values('Ratio', ascending=False)
m_u = most_underrated.iloc[:10,:]

