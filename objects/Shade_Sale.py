import json, re
from datetime import datetime

def read_jsons(list_dir, spec_dir, item_num):
    with open(f'{list_dir}/{item_num}_listing.json', "r") as read_file:
        listing = json.load(read_file)
    with open('%s/%s_specs.json' % (spec_dir, item_num), "r") as read_file:
        specs = json.load(read_file)
    return {'listing': listing, 'specs': specs}

class Shade_Sale:
    def __init__(self, list_dir, spec_dir, item_num=None):
        
        # UID
        self.id = item_num
        self.__body = read_jsons(list_dir, spec_dir, "%s" % (item_num.split('_')[0]))
        
        # ULTIMATE PRICE - SCALAR DEPENDENT VAR
        self.price = self.__body['specs']['ConvertedCurrentPrice']['Value']

        # INITIAL PRICE - SCALAR
        self.initial_price = float(self.__body['listing']['sellingStatus'][0]['convertedCurrentPrice'][0]['__value__'])
        
        # SHIPPING COST - SCALAR INDEPENDENT VARIABLE ("IV")
        self.price_shipping = self.__body['listing']['shippingInfo'][0].get('shippingServiceCost')
        if self.price_shipping:
            self.price_shipping = float(self.__body['listing']['shippingInfo'][0]['shippingServiceCost'][0]['__value__'])
        else: self.price_shipping = float(0)
        
        # TITLE - UNSTRUCTURED TEXT DATA FOR NLP ANALYSIS - STUFF WILL BE APPENDED TO THIS LATER
        self.title = self.__body['listing']['title'][0]
        self.len_title = len(self.__body['listing']['title'][0])
        
        # MARKET COUNTRY (ONE-HOT CATEGORICAL VARIABLE)
        self.market = self.__body['listing']['globalId'][0]
        
        # PIC URL (JUST FOR FUNSIES)
        if self.__body['listing'].get('galleryURL'):
            self.pic = self.__body['listing']['galleryURL'][0]
        else: self.pic = None
        
        # BIG PIC URL (JUST FOR FUNSIES)
        if self.__body['listing'].get("pictureURLLarge"):
            self.pic_big = self.__body['listing']['pictureURLLarge'][0]
        else:
            self.pic_big = None
        
        # ITEM URL (JUST FOR FUNSIES)
        self.url = self.__body['listing']['viewItemURL'][0]
        
        # AUTOPAY (ONE-HOT CATEGORICAL VARIABLE)
        self.autopay = self.__body['listing']['autoPay'][0] == 'true'
        
        # SELLER COUNTRY (ONE-HOT CATEGORICAL VAR)
        self.country_seller = self.__body['listing']['country'][0]
        if not self.country_seller:
            self.country_seller = 'NOT_LISTED'
            
        # RETURNS OFFERED (ONE-HOT)
        self.returns = self.__body['listing']['returnsAccepted'][0] == 'true'
        
        # LISTING INFO (FOR FURTHER PARSING BELOW)
        self.listing_info = self.__body['listing']['listingInfo'][0]
        
        # LISTING TYPE - 1-HOT
        self.listing_type = self.listing_info['listingType'][0]
        
        ################# SHIPPING STUFF #################
        
        # SHIPPING TYPE - 1-HOT CATEGORICAL
        self.ship_type = self.__body['listing']['shippingInfo'][0]['shippingType'][0]
        
        # SHIPPING EXPEDITE - 1-HOT
        self.ship_expedite = self.__body['listing']['shippingInfo'][0]['expeditedShipping'][0] == 'true'
        
        # SHIPPING HANDLING TIME - 1-HOT
        if self.__body['listing']['shippingInfo'][0].get('handlingTime'):
            self.ship_handling_time = int(self.__body['listing']['shippingInfo'][0]['handlingTime'][0])
        else: 
            self.ship_handling_time = 0
        
        # SELLER ZIP CODE (JUST FOR FUNSIES)
        self.zip = self.__body['listing'].get('postalCode')
        
        ################# TEMPORAL STUFF #################
        
        # START AND END TIMES - 1-HOT THE HOUR OF DAY
        self.start_time = datetime.strptime(self.listing_info['startTime'][0], 
                                            "%Y-%m-%dT%H:%M:%S.%fZ")
        self.end_time = datetime.strptime(self.listing_info['endTime'][0], 
                                            "%Y-%m-%dT%H:%M:%S.%fZ")
        
        # START AND END WEEKDAYS - 1-HOT DAY OF WEEK
        self.start_weekday = self.start_time.weekday()
        self.end_weekday = self.end_time.weekday()
        
        # DURATION OF AUCTION IN HOURS - SCALAR NUMBER OF HOURS
        self.__duration = (self.end_time - self.start_time)
        self.duration = round(float(self.__duration.days*24) + float(self.__duration.seconds/60/60), 2)
        
        # RETURN WINDOW DURATION - 1-HOT ENCODE
        if self.returns:
            if self.__body['specs']['ReturnPolicy'].get('ReturnsWithin'):
                self.returns_time = int(self.__body['specs']['ReturnPolicy'].get('ReturnsWithin').split()[0])
        else: self.returns_time = 0
        
        ################# ITEM SPECIFICS: #################
        
        # TEXT DESCRIPTION - UNSTRUCTURED - NLP THIS
        if self.__body['specs'].get('Description'):
            self.description = self.__body['specs']['Description']
        else:
            self.description = 'sunglasses'
            # ALSO ONE-HOT ENCODE WHETHER DESCRIPTION EXISTS OR NOT
            
        # SUBTITLE - ONE-HOT ENCODE WHETHER THIS EXISTS OR NOT
        self.subtitle = self.__body['specs'].get('Subtitle')
        # ALSO APPEND IT TO THE ITEM TITLE FOR NLP
        
        # CONDITION DESCRIPTION - ONE-HOT ENCODE WHETHER THIS EXISTS OR NOT
        self.condition_description = self.__body['specs'].get('ConditionDescription')
        # ALSO APPEND TO ITEM TITLE FOR NLP
        
        # NUMBER OF PICS - BIN THESE INTO LESS THAN / GREATER THAN CATEGORIES
        self.pic_quantity = len(self.__body['specs']['PictureURL'])

        # SELLER FEEDBACK SCORE - BIN THIS INTO CATEGORIES
        if self.__body['specs']['Seller'].get('FeedbackScore'):
            self.seller_feedback_score = float(self.__body['specs']['Seller']['FeedbackScore'])
        else: 
            self.seller_feedback_score = None
        
        # SELLER POSITIVE PERCENT - BIN THIS INTO CATEGORIES
        self.seller_positive_percent = float(self.__body['specs']['Seller']['PositiveFeedbackPercent'])
        
        # BEST OFFER ENABLED - ONE-HOT ENCODE
        self.best_offer_enabled = self.__body['specs']['BestOfferEnabled']
        
        ################# NITTY GRITTIES #################
        
        # GET THE ITEM SPECIFICS FOR FURTHER PARSING
        if self.__body['specs'].get('ItemSpecifics'):
            self.__attrs = {prop['Name']:prop['Value'][0] 
                            for prop in self.__body['specs']['ItemSpecifics']['NameValueList']}
            
            # BRAND / MAKE OF ITEM - 1-HOT ENCODE
            # 1-HOT ENCODE THE MERE EXISTENCE OF THIS VALUE
            # TOSS INTO NLP STEW

            self.brand = self.__attrs.get('Brand')
            if self.brand:
                self.brand = self.brand.upper()
                if 'RAY BAN' in self.brand or 'RAYBAN' in self.brand:
                    self.brand = 'RAY-BAN'
                if self.brand not in ['OAKLEY', 'RAY-BAN', 'OTHER', 'MAUI JIM', 'COSTA DEL MAR',
                                        'PERSOL', 'CARTIER', 'GUCCI']:
                    self.brand = "OTHER"
            else: self.brand = "UNLISTED"
                
            # MATERIAL
            # 1-HOT EXISTENCE OF THIS VARIABLE
            # TOSS IT INTO THE NLP STEW
            self.material = self.__attrs.get('Body Material')
            if self.material:
                self.material = self.material.upper()
                if "MAHOGANY" in self.material:
                    self.material = "MAHOGANY"
                if "ROSEWOOD" in self.material:
                    self.material = "ROSEWOOD"
                if "ALDER" in self.material:
                    self.material = "ALDER"
                if "ASH" in self.material:
                    self.material = "ASH"
                if "MAPLE" in self.material:
                    self.material = "MAPLE"
            
            
            # COUNTRY OF MANUFACTURE - 1-HOT THIS
            self.country_manufacture = self.__attrs.get('Country/Region of Manufacture')
            if self.country_manufacture:
                self.country_manufacture = self.country_manufacture.upper()
                if self.country_manufacture == 'KOREA, REPUBLIC OF':
                    self.country_manufacture = 'KOREA'
                if self.country_manufacture == 'UNITED STATES':
                    self.country_manufacture = 'USA'
                if self.country_manufacture not in 'ITALY USA JAPAN'.split():
                    self.country_manufacture = 'OTHER'
            else: self.country_manufacture = "NOT_LISTED"
                # if self.country_manufacture not in ['USA', 'JAPAN',\
                #                                     'MEXICO', 'KOREA',\
                #                                     'CHINA','INDONESIA']:
                #     self.country_manufacture = 'OTHER'
            
            
            # BODY TYPE - 1-HOT CATEGORICAL
            self.body_type = self.__attrs.get('Body Type')
            if self.body_type:
                self.body_type = self.body_type.upper()
                if self.body_type == "HOLLOW BODY" or self.body_type == "HOLLOWBODY" or self.body_type == "HOLLOW-BODY":
                    self.body_type = "HOLLOW"
                if self.body_type == "SOLID BODY" or self.body_type == "SOLIDBODY" or self.body_type == "SOLID-BODY":
                    self.body_type = "SOLID"
                if "STRAT" in self.body_type:
                    self.body_type = "STRAT"
                if self.body_type not in ['SOLID', 'SEMI-HOLLOW', 'HOLLOW', 'CLASSICAL']:
                    self.body_type = "OTHER"
            
            # BODY COLOR - DUMP (+ "COLORED" for bigram) INTO NLP WITH DESCRIPTION
            # ONE-HOT ENCODE THE EXISTENCE OF THIS VARIABLE
            self.color = self.__attrs.get('Body Color')
            if self.color:
                self.color = self.color.upper()
                if "CHERRY" in self.color:
                    self.color = "RED"
                if "SUNBURST" in self.color:
                    self.color = "SUNBURST"
                if "BURST" in self.color:
                    self.color = "SUNBURST"
                if "MAHOGANY" in self.color:
                    self.color = "NATURAL"
                if "BLUE" in self.color:
                    self.color = "BLUE"
                if "TURQUOISE" in self.color:
                    self.color = "BLUE"
                if "TEAL" in self.color:
                    self.color = "BLUE"
                if "RED" in self.color:
                    self.color = "RED"
                if "BLACK" in self.color:
                    self.color = "BLACK"
                if "EBONY" in self.color:
                    self.color = "BLACK"
                if "WHITE" in self.color:
                    self.color = "WHITE"
                if "GREEN" in self.color:
                    self.color = "GREEN"
                if "NATURAL" in self.color:
                    self.color = "NATURAL"
                if "BLONDE" in self.color:
                    self.color = "NATURAL"
                if "BLOND" in self.color:
                    self.color = "NATURAL"
                if "BEIGE" in self.color:
                    self.color = "NATURAL"
                if "MAPLE" in self.color:
                    self.color = "NATURAL"
                if "BUTTERSCOTCH" in self.color:
                    self.color = "NATURAL"
                if "WALNUT" in self.color:
                    self.color = "NATURAL"
                if "TOBACCO" in self.color:
                    self.color = "NATURAL"
                if "BROWN" in self.color:
                    self.color = "NATURAL"
                if "CREAM" in self.color:
                    self.color = "WHITE"
                if "GOLD" in self.color:
                    self.color = "YELLOW"
                if "YELLOW" in self.color:
                    self.color = "YELLOW"
                if "FIREGLO" in self.color:
                    self.color = "RED"
                if "WINE" in self.color:
                    self.color = "RED"
                if "BURGANDY" in self.color:
                    self.color = "RED"
                if "BURGUNDY" in self.color:
                    self.color = "RED"
                if "MULTI-COLOR" in self.color:
                    self.color = "MULTICOLOR"
                if "AMBER" in self.color:
                    self.color = "YELLOW"
                if "WOOD" in self.color:
                    self.color = "NATURAL"
                if "COPPER" in self.color:
                    self.color = "RED"
                if "PEWTER" in self.color:
                    self.color = "GRAY"
                if "GRAY" in self.color:
                    self.color = "GRAY"
                if self.color not in ['BLACK', 'RED', 'SUNBURST', 'WHITE', 'NATURAL', 'BLUE', 'YELLOW',
                                       'GREEN']:
                    self.color = "OTHER"
            else: self.color = None
        
        # INITIALIZING VARIABLES THAT DIDN'T GET ASSIGNED VALUES
        else:
            self.color = self.brand = self.model = 'UNLISTED'
            self.material = self.right_left_handed = self.country_manufacture = None
            self.body_type = self.string_config = self.listing_type = None    