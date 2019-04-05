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

def page_of_listings_to_mongo(page):
    listings = db.listings

    if listings.count_documents({}) == 0:
        listings.create_index([('itemId', pymongo.ASCENDING)], unique=True)

    i = 0
    for listing in page:
        try:
            listings.insert_one(listing)
        except pymongo.errors.DuplicateKeyError:
            i += 1
            continue
    print(f'Omitted {i} dupe items')
    pass

def item_spec_to_mongo(spec):
    '''Writes one page of Axe Specs to mongodb'''
    specs = db.specs
    if specs.count_documents({}) == 0:
        specs.create_index([('ItemID', pymongo.ASCENDING)], unique=True)
    try:
        specs.insert_one(spec)
    except pymongo.errors.DuplicateKeyError:
        print('Skip duplicate')
        pass
    return 0

def spam_the_api_mongo(start_page, stop_page, fetch_function):
    # existing_files = [name.split('_')[0] for name in os.listdir('data/specs/') if not name.startswith('.')] # Ignore .DS_Store
    j = 0
    k = 0    
    '''Spams the eBay API for pages of item DATA'''
    for i in range(start_page+1, stop_page+1):
        page = fetch_function(i)
        if page != None:
            page_of_listings_to_mongo(page)
            for response_item in page:
                k += 1
                if response_item['itemId'][0] not in [i['ItemID'] for i in db.specs.find({})]:
                    j += 1
                    print(f'Get spec for page {i} item {k}')
                    item_spec_to_mongo(get_specs(response_item['itemId'][0]))
                else: print(f'Already have spec for page {i} item {k}')
    print(f'\nChecked {k} items')
    print(f'\nFetched specs for {j} new items')
    return 0

# Again, you only get 5k API calls per day.
# spam_the_api(0,110, find_completed_auction)
# spam_the_api(0, 110, find_completed_auction_BIN)