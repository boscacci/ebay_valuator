#!/usr/bin/env python
# coding: utf-8

import json, os, requests

import pymongo
from pymongo import MongoClient
client = MongoClient()
db = client.ebay_val

from key import API_KEY
API_KEY = API_KEY # Enter your API Key/"App ID" Here. Mine was 40 chars long.

# FILM_CAMS = '15230'
# ELEC_GUITARS = '33034' # Category code for electric guitars on eBay.
# WATCHES = '14324'
MENS_SUNGLASSES = '79720'

USA = 'EBAY-US' # USA Marketplace only. API seems to start returning stuff from other markets eventually anyhow
USED = '3000' # Just the code for item condition "used". This is the focus of this project.
AUCTIONWITHBIN = "AuctionWithBIN"
AUCTION = "Auction"

ITEM_FILTER_0 = f'itemFilter(0).name=Condition&itemFilter(0).value={USED}' # Only used guitars
ITEM_FILTER_1 = f'itemFilter(1).name=HideDuplicateItems&itemFilter(1).value=true' # No duplicate listings
ITEM_FILTER_2 = f'itemFilter(2).name=MinPrice&itemFilter(2).value=9' # Only items that sold for > this value
ITEM_FILTER_3 = f'itemFilter(3).name=MaxQuantity&itemFilter(3).value=1' # No lots or batch sales. One item at a time
ITEM_FILTER_4 = f'itemFilter(4).name=SoldItemsOnly&itemFilter(4).value=true' # Only looking at listings that sold.
ITEM_FILTER_5 = f'itemFilter(5).name=ListingType&itemFilter(5).value={AUCTION}' # Select sale type.
ITEM_FILTER_6 = f'itemFilter(6).name=MaxPrice&itemFilter(6).value=3000' # Only items that sold for < this value
ITEM_FILTER_7 = f'itemFilter(7).name=listingType&itemFilter(7).value={AUCTIONWITHBIN}' # Select sale type.

FIND_COMPLETED = 'findCompletedItems' # This is the eBay API endpoint service we'll be querying.

def find_completed_auction(PAGE):
    '''Give it a page num, returns a list of json-style listings'''
    r = requests.get(f'https://svcs.ebay.com/services/search/FindingService/v1?'
                 f'OPERATION-NAME={FIND_COMPLETED}&'
                 f'X-EBAY-SOA-SECURITY-APPNAME={API_KEY}&'
                 f'RESPONSE-DATA-FORMAT=JSON&' # This value can be altered if you're not into JSON responses
                 f'REST-PAYLOAD&'
                 f'GLOBAL-ID={USA}&' # seems to prioritize the value you enter but returns other stuff too
                 f'categoryId={MENS_SUNGLASSES}&' # Product category goes here
                 f'descriptionSearch=true&' # More verbose responses
                 f'{ITEM_FILTER_0}&' # USED
                 f'{ITEM_FILTER_1}&' # NO DUPES
                 f'{ITEM_FILTER_2}&' # MINPRICE
                 f'{ITEM_FILTER_3}&' # NO LOTS
                 f'{ITEM_FILTER_4}&' # ONLY SOLD
                 f'{ITEM_FILTER_5}&' # AUCTIONS ONLY
                 f'{ITEM_FILTER_6}&' # MAX PRICE
                 f'paginationInput.pageNumber={str(PAGE)}&' # value to be looped through when collecting lotsa data
                 f'outputSelector=PictureURLLarge') # Why not grab the thumbnail URLs too. Could be cool
    if r.json()['findCompletedItemsResponse'][0].get('searchResult'):
        return r.json()['findCompletedItemsResponse'][0]['searchResult'][0]['item']
    else:
        return None

# For BUY IT NOW:
def find_completed_auction_BIN(PAGE):
    '''Give it a page num, returns a list of json-style listings'''
    r = requests.get(f'https://svcs.ebay.com/services/search/FindingService/v1?'
                 f'OPERATION-NAME={FIND_COMPLETED}&'
                 f'X-EBAY-SOA-SECURITY-APPNAME={API_KEY}&'
                 f'RESPONSE-DATA-FORMAT=JSON&' # This value can be altered if you're not into JSON responses
                 f'REST-PAYLOAD&'
                 f'GLOBAL-ID={USA}&' # seems to prioritize the value you enter but returns other stuff too
                 f'categoryId={MENS_SUNGLASSES}&' # Product category goes here
                 f'descriptionSearch=true&' # More verbose responses
                 f'{ITEM_FILTER_0}&' # USED
                 f'{ITEM_FILTER_1}&' # NO DUPES
                 f'{ITEM_FILTER_2}&' # MINPRICE
                 f'{ITEM_FILTER_3}&' # NO LOTS
                 f'{ITEM_FILTER_4}&' # ONLY SOLD
                 f'{ITEM_FILTER_7}&' # BUY IT NOW ONLY
                 f'paginationInput.pageNumber={str(PAGE)}&' # value to be looped through when collecting lotsa data
                 f'outputSelector=PictureURLLarge') # Why not grab the thumbnail URLs too. Could be cool
    if r.json()['findCompletedItemsResponse'][0].get('searchResult'):
        # This returns the whole page as a list of dicts:
        return r.json()['findCompletedItemsResponse'][0]['searchResult'][0]['item']
    else:
        return None


def get_specs(ITEM_ID):
    '''Return the specifics of a single eBay auction. String input.'''
    r2 = requests.get('http://open.api.ebay.com/shopping?'
                    f'callname=GetSingleItem&'
                    f'responseencoding=JSON&'
                    f'appid={API_KEY}&'
                    f'siteid=0&' # USA Store
                    f'version=967&' # What is this?
                    f'ItemID={ITEM_ID}&' # Assigned above
                    f'IncludeSelector=Details,ItemSpecifics,TextDescription')
    try:
        return r2.json()['Item']
    except KeyError:
        pass

# Seems like there's some variability between items when it comes to item specifics field.
# ### Persisting some Data for Analysis

# Just write first page of listing results to .json files:
def persist_page_to_json(PAGE):
    '''Saves a page of JSON responses to one json per sale / item'''
    for i in range(len(PAGE)):
        with open("data/listings/%s_listing.json" % (PAGE[i]['itemId'][0]), 'w') as f:  # writing JSON object
            json.dump(PAGE[i], f)

# Now write one page of details to a JSON:
def persist_spec_to_json(spec):
    '''Writes one page of Axe Specs to one json'''
    try:
        with open("data/specs/%s_specs.json" % (spec['ItemID']), 'w') as f:  # writing JSON object
            json.dump(spec, f)
    except TypeError:
        pass
    pass


def page_of_listings_to_mongo(page):
    listings = db.listings
    for listing in page:
        try:
            listings.insert_one(listing)
        except pymongo.errors.DuplicateKeyError:
            continue
    pass

def spam_the_api_mongo(start_page, stop_page, fetch_function):
    # existing_files = [name.split('_')[0] for name in os.listdir('data/specs/') if not name.startswith('.')] # Ignore .DS_Store
    j = 0
    k = 0    
    '''Spams the eBay API for pages of item DATA'''
    for i in range(start_page+1, stop_page+1):
        page = fetch_function(i)
        if page != None:
            # import pdb; pdb.set_trace()
            page_of_listings_to_mongo(page)
    pass

# Okay, careful, this is where we start to hammer the eBay API a little bit.
def spam_the_api(start_page, stop_page, fetch_function):
    existing_files = [name.split('_')[0] for name in os.listdir('data/specs/') if not name.startswith('.')] # Ignore .DS_Store
    
    j = 0
    k = 0
    
    '''Spams the eBay API for pages of item DATA'''
    
    for i in range(start_page+1, stop_page+1):
        page = fetch_function(i)
        if page != None:
            persist_page_to_json(page)
            for item in page:
                k += 1
                if item['itemId'][0] not in existing_files:
                    j += 1
                    print(f'Get page {i} item {k}')
                    persist_spec_to_json(get_specs(item['itemId'][0]))
                else:
                    print(f'Skip page {i} item {k}')    
    print(f'\nChecked {k} items')
    print(f'\nGot {j} new items')


# Again, you only get 5k API calls per day.
# spam_the_api(0,110, find_completed_auction)
# spam_the_api(0, 110, find_completed_auction_BIN)