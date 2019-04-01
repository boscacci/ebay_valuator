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

from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, 'objects')
sys.path.insert(0, 'pickles')

# Trawl for Prospective Deals

address = input("Enter recipient email address: ")
days_ahead = int(input("Enter how many days ahead you want to scrape for: "))

API_KEY = API_KEY # Enter your API Key/"App ID" Here. Mine was 40 chars long.

FIND_ADVANCED = "findItemsAdvanced" # This is the eBay API endpoint service we'll be querying.
ELEC_GUITARS = '33034'
USED = '3000'
AUCTION = "Auction"
AUCTIONWITHBIN = "AuctionWithBIN"


ct = datetime.utcnow()
now = datetime.now()

two_days_from_now = now + timedelta(days=days_ahead)

utc_ct = f'{ct.year}-'
if len(str(ct.month)) < 2:
    utc_ct += '0'
utc_ct += f'{ct.month}-'

if int(ct.day) + days_ahead < 10:
    utc_ct += '0'

utc_ct += str(ct.day + days_ahead) + 'T'

if len(str(ct.hour)) < 2:
    utc_ct += '0'
utc_ct += f'{ct.hour}:'
if len(str(ct.minute)) < 2:
    utc_ct += '0'
utc_ct += f'{ct.minute}:'
if len(str(ct.second)) < 2:
    utc_ct += '0'
utc_ct += f'{ct.second}.'
if len(str(ct.microsecond)) < 2:    
    utc_ct += '0'
utc_ct += str(ct.microsecond)[:3] + 'Z'

ITEM_FILTER_0 = f'itemFilter(0).name=Condition&itemFilter(0).value={USED}' # Only used guitars
ITEM_FILTER_1 = f'itemFilter(1).name=HideDuplicateItems&itemFilter(1).value=true' # No duplicate listings
ITEM_FILTER_2 = f'itemFilter(2).name=MinPrice&itemFilter(2).value=1' # Only items that sell for > this value
ITEM_FILTER_3 = f'itemFilter(3).name=MaxQuantity&itemFilter(3).value=1' # No lots or batch sales. One item at a time
ITEM_FILTER_4 = f'itemFilter(4).name=MaxPrice&itemFilter(4).value=310' # Only items that sold for < this value
ITEM_FILTER_5 = f'itemFilter(5).name=EndTimeTo&itemFilter(5).value={utc_ct}' # Only ending soonish

def find_current_auctions(PAGE, keywords):
    '''Make a request to the eBay API and return the JSON text of this page number'''
    r = requests.get(
                 f'https://svcs.ebay.com/services/search/FindingService/v1?'
                 f'OPERATION-NAME={FIND_ADVANCED}&'
                 f'X-EBAY-SOA-SECURITY-APPNAME={API_KEY}&'
                 f'RESPONSE-DATA-FORMAT=JSON&'
                 f'REST-PAYLOAD&'
                 f'categoryId={ELEC_GUITARS}&'
                 f'descriptionSearch=true&'
                 f'{ITEM_FILTER_0}&' # USED
                 f'{ITEM_FILTER_1}&' # NO DUPES
                 f'{ITEM_FILTER_2}&' # MINPRICE
                 f'{ITEM_FILTER_3}&' # NO LOTS
                 f'{ITEM_FILTER_4}&' # MAX PRICE
                 f'{ITEM_FILTER_5}&' # END TIME
                 f'keywords={keywords}&'
                 f'paginationInput.pageNumber={str(PAGE)}') # value to be looped through when collecting lotsa data
    if r.json()['findItemsAdvancedResponse'][0].get('searchResult'):
        return r.json()['findItemsAdvancedResponse'][0]['searchResult'][0]['item']
    else:
        return None

def get_specs(ITEM_ID):
    '''Return the specifics of a single eBay auction. String input.'''
    r = requests.get('http://open.api.ebay.com/shopping?'
                    f'callname=GetSingleItem&'
                    f'responseencoding=JSON&'
                    f'appid={API_KEY}&'
                    f'version=967&' # What is this?
                    f'ItemID={ITEM_ID}&'
                    f'IncludeSelector=Details,ItemSpecifics,TextDescription')
    try:
        return r.json()['Item']
    except KeyError:
        pass


def trawl_for_guitars(start_page, stop_page, fetch_function, keywords):
    '''Spams the eBay API for pages of AXE DATA'''
    j = 0
    k = 0
    existing_guitar_ids = []
    listings = []
    
    for i in range(start_page+1, stop_page+1):
        page = fetch_function(i, keywords)
        if page:
            for axe in page:
                k += 1
                if axe['itemId'][0] not in existing_guitar_ids:
                    existing_guitar_ids.append(axe['itemId'][0])
                    j += 1
                    print(f'Get {j}')
                    listings.append({'listing': axe,
                                     'specs': get_specs(axe['itemId'][0])})
                else:
                    print('Skip')
    
    print(f'\nChecked {k} guitars')
    print(f'\nGot {j} new guitars')
    
    return listings

prospects = []


print(stars)
print('Trawling for goodies on the online:')

# See this for eBay keyword formatting: https://developer.ebay.com/Devzone/finding/Concepts/FindingAPIGuide.html#usekeywords
# URL Formatting can be found here: https://www.freeformatter.com/url-encoder.html
prospects.extend(trawl_for_guitars(0,3,find_current_auctions,'american+-%28squier%2Csquire%2Cepiphone%2Cepi%29'))
prospects.extend(trawl_for_guitars(0,3,find_current_auctions,'fender+-%28squier%2Csquire%2Cepiphone%2Cepi%29'))
prospects.extend(trawl_for_guitars(0,3,find_current_auctions, 'gibson+-%28squier%2Csquire%2Cepiphone%2Cepi%29'))
prospects.extend(trawl_for_guitars(0,3,find_current_auctions, 'ibanez+-%28squier%2Csquire%2Cepiphone%2Cepi%29'))
prospects.extend(trawl_for_guitars(0,3,find_current_auctions, 'schecter+-%28squier%2Csquire%2Cepiphone%2Cepi%29'))
prospects.extend(trawl_for_guitars(0,3,find_current_auctions, 'esp+-%28squier%2Csquire%2Cepiphone%2Cepi%29'))
prospects.extend(trawl_for_guitars(0,3,find_current_auctions, 'jackson+-%28squier%2Csquire%2Cepiphone%2Cepi%29'))
prospects.extend(trawl_for_guitars(0,3,find_current_auctions, 'prs+-%28squier%2Csquire%2Cepiphone%2Cepi%29'))
prospects.extend(trawl_for_guitars(0,3,find_current_auctions, 'gretsch+-%28squier%2Csquire%2Cepiphone%2Cepi%29'))
prospects.extend(trawl_for_guitars(0,3,find_current_auctions, 'japanese+-%28squier%2Csquire%2Cepiphone%2Cepi%29'))


print(stars)
print('De-Serializing what we got from eBay:')

from Axe_Object_memory import Axe
# from Axe_Object import Axe

guitars = []
for prospect in prospects:
    if prospect.get('specs'):
        try:
            this_axe = Axe(prospect['listing'],prospect['specs'])
            if "LOT OF" not in this_axe.title.upper() and this_axe.price > 90 and this_axe.price < 800\
        and "TREMOLO" not in this_axe.title.upper():
                if this_axe.string_config and this_axe.string_config < 5:
                    continue
                if this_axe.year and this_axe.year > 2019:
                    continue
                if this_axe.end_time > two_days_from_now or this_axe.end_time < now:
                    continue
                guitars.append(this_axe)
        except ValueError:
            print("skip")
            pass
    else:
        print("skip")
        pass

# print(stars)
# print('Importing JSON Guitars')

# file_names = [name for name in os.listdir('data/axe_specs/') if not name.startswith('.')] # Ignores hidden files on mac

# guitars = []
# for filename in file_names:
#     try:
#         this_axe = Axe('data/axe_listings', 'data/axe_specs', filename)
#         if "LOT OF" not in this_axe.title.upper() and this_axe.price > 110 and this_axe.price < 890\
#         and "TREMOLO" not in this_axe.title.upper():
#             if this_axe.string_config and this_axe.string_config < 5:
#                 continue
#             if this_axe.market != 'EBAY-US':
#                 continue
#             if this_axe.year and this_axe.year > 2019:
#                 continue
#             guitars.append(this_axe)
#     except ValueError:
#         pass

print(stars)
print('More organizing data:')

# Properties
title_lengths       = pd.Series([guitar.len_title for guitar in guitars], name = 'title_lengths')
auction_duration    = pd.Series(np.full(len(guitars),7*24), name = 'auction_duration')
shipping_charged    = pd.Series(np.full(len(guitars),0), name = 'shipping_charged')          
seller_country_US   = pd.Series([1 for i in range(len(guitars))], name = 'seller_country_US')
autopay             = pd.Series([False for guitar in range(len(guitars))], name = 'autopay')
returns             = pd.Series([False for guitar in range(len(guitars))], name = 'returns')
listing_type_FixedPrice = pd.Series([True for guitar in range(len(guitars))], name = 'listing_type_FixedPrice')
ship_type_Free      = pd.Series([1 for guitar in range(len(guitars))], name = 'ship_type_Free')
ship_expedite       = pd.Series([0 for guitar in range(len(guitars))], name = 'ship_expedite')
start_hour          = pd.Series([1 for guitar in range(len(guitars))], name = 'start_hour_(7.667, 11.5]')
end_hour            = pd.Series([1 for guitar in range(len(guitars))], name = 'end_hour_(7.667, 11.5]')
start_weekday_6     = pd.Series([6 for guitar in range(len(guitars))], name = 'start_weekday_6')
end_weekday_6       = pd.Series([6 for guitar in range(len(guitars))], name = 'end_weekday_6')
returns_time      = pd.Series([0 for guitar in range(len(guitars))], name = 'returns_time')
num_pics            = pd.Series([12 for guitar in range(len(guitars))], name = 'num_pics')
best_offer_enabled  = pd.Series([True for guitar in range(len(guitars))], name = 'best_offer_enabled')
ship_handling_time_2= pd.Series([1 for guitar in range(len(guitars))], name = 'ship_handling_time_2')
seller_positive_percent = pd.Series([1 for guitar in range(len(guitars))], name = 'seller_positive_percent_(99.5, 111.0]')
brand = pd.Series([guitar.brand for guitar in guitars], name = "brand")
body_type           = pd.Series([guitar.body_type for guitar in guitars], name = "body_type")
color               = pd.Series([guitar.color for guitar in guitars], name = "color")
right_left_handed   = pd.Series([guitar.right_left_handed for guitar in guitars], name = "right_left_handed")
string_config       = pd.cut(pd.Series([guitar.string_config for guitar in guitars], name = "string_config"),
                       [0,5,6,11,20])
country_manufacture = pd.Series([guitar.country_manufacture for guitar in guitars], name = "country_manufacture")
model_year = pd.cut(pd.Series([guitar.year for guitar in guitars], name = "model_year"), [1700,1975,1990,1995,2000,2005,2007,2010,2011,2012,2013,2015])

X_dummies = pd.concat([title_lengths, brand, color, country_manufacture, right_left_handed, best_offer_enabled, shipping_charged, 
               returns, returns_time, autopay, seller_country_US, ship_handling_time_2, listing_type_FixedPrice, ship_expedite,
               ship_type_Free, num_pics, auction_duration, start_hour, end_hour, start_weekday_6, end_weekday_6, 
               seller_positive_percent, model_year, body_type, string_config],
              axis = 1)

X_nontext = pd.get_dummies(X_dummies, drop_first=True)

# print(stars)
# print('Prepping new data to feed into lasso:')

stemmer = SnowballStemmer("english")

stopwords_list = stopwords.words('english') + list(string.punctuation)
stopwords_list += ["''", '""', '...', '``', ",", ".", ":", "'s", "--","’"]

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

def process_doc(doc):
    stopwords_removed = ''
    tokens = nltk.word_tokenize(doc)
    for i in range(len(tokens)):
        if tokens[i].lower() not in stopwords_list and tokens[i] not in string.punctuation:
            stopwords_removed += stemmer.stem(tokens[i]) + ' '
    return stopwords_removed

print(stars)
print('Analyzing text of new prospects:')

raw_corpus = [assemble_guitar_document(guitar).lower() for guitar in guitars]
processed_text = pd.Series(list(map(process_doc, raw_corpus)), name = 'text')


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
        filler = pd.Series(np.full(len(guitars),0), name=col)
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

bxcx_lam = .3
y_preds_inv = inv_boxcox(y_preds, bxcx_lam)

bids = []
for guitar in guitars:
    if guitar._Axe__body['listing']['sellingStatus'][0].get('bidCount'):
        bids.append(guitar._Axe__body['listing']['sellingStatus'][0]['bidCount'][0])
    else:
        bids.append('0')

predicted_df = pd.concat([pd.Series(y_preds_inv), pd.Series([guitar.initial_price + guitar.price_shipping for guitar in guitars]),
                          pd.Series(y_preds_inv) / pd.Series([guitar.initial_price for guitar in guitars]),
                          pd.Series([guitar.title for guitar in guitars]), 
                          pd.Series([guitar.url for guitar in guitars]), 
                         pd.Series([guitar.pic for guitar in guitars]),
                         pd.Series(bids)],
                         axis=1)

predicted_df.columns = ['Estimate', 'Price', 'Ratio','Title','Link', 'Pic', 'Bids']

highest_value = predicted_df.sort_values('Estimate', ascending=False)
hv_10 = highest_value.iloc[:10,:]

most_underrated = predicted_df.sort_values('Ratio', ascending=False)
m_u = most_underrated.iloc[:10,:]

print(stars)
print('Formatting an email:')

email = '\n<h3>10 Highest Potential Value:</h3>\n'
email += "*******************************\n"
    
for i in range(10):
    email += f"<a href = {hv_10['Link'].values[i]}>"
    email += f"\n{hv_10['Title'].values[i]}\n\n"
    email += f"<img src = {hv_10.Pic.values[i]}></img></a>\n"
    email += f"\nCurrent Price: ${hv_10['Price'].values[i]}"
    email += f"\nBids: {m_u['Bids'].values[i]}\n\n"
    email += "*******************************\n"
    
email += "<h3>10 Most Underrated:</h3>\n"

email += "*******************************\n"

for i in range(10):
    email += f"<a href = {m_u['Link'].values[i]}>"
    email += f"\n{m_u['Title'].values[i]}\n\n:"
    email += f"<img src = {m_u['Pic'].values[i]}></img></a>\n"
    email += f"\nCurrent Price: ${m_u['Price'].values[i]}"
    email += f"\nBids: {m_u['Bids'].values[i]}\n\n"
    email += "*******************************\n"

yag = yagmail.SMTP("gu1tarb1trag3@gmail.com")
yag.send(
    to=address,
    subject=f"GuitArbitrage - Auctions Ending Within {days_ahead} Day(s)",
    contents=email)

print(f"Summary sent to {address.split('@')[0]}. Happy hunting")
