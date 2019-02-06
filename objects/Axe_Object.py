import json, re
from datetime import datetime

def read_axe_jsons(list_dir, spec_dir, axe_num):
    with open('%s/%s' % (list_dir,axe_num), "r") as read_file:
        listing = json.load(read_file)
    with open('%s/%s' % (spec_dir, axe_num), "r") as read_file:
        specs = json.load(read_file)
    return {'listing': listing, 'specs': specs}

class Axe:
    def __init__(self, list_dir, spec_dir, axe_num=None):
        
        # UID
        self.id = axe_num
        self.__body = read_axe_jsons(list_dir, spec_dir, "%s" % (axe_num))
        
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
        if self.country_seller not in ['US', 'JP', 'CA', 'GB']:
            self.country_seller = 'OTHER'
            
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
            self.description = None
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
            
            # BRAND / MAKE OF GUITAR - 1-HOT ENCODE
            # 1-HOT ENCODE THE MERE EXISTENCE OF THIS VALUE
            # TOSS INTO NLP STEW
            self.brand = self.__attrs.get('Brand')
            if self.brand:
                self.brand = self.brand.upper()
                if "GIBSON" in self.brand:
                    self.brand = "GIBSON"
                if "FENDER" in self.brand:
                    self.brand = "FENDER"
                if "SCHECTER" in self.brand:
                    self.brand = "SCHECTER"
                if "MUSIC MAN" in self.brand or "MUSICMAN" in self.brand:
                    self.brand = "MUSIC MAN"
                if "ESP" in self.brand:
                    self.brand = "ESP"
                if "YAMAHA" in self.brand:
                    self.brand = "YAMAHA"
                if "RICKENBACKER" in self.brand:
                    self.brand = "RICKENBACKER"
                if "WASHBURN" in self.brand:
                    self.brand = "WASHBURN"
                if "PRS" in self.brand:
                    self.brand = "PRS"
                if "PAUL REED SMITH" in self.brand:
                    self.brand = "PRS"
                if "GRETSCH" in self.brand:
                    self.brand = "GRETSCH"
                if self.brand not in ['FENDER', 'GIBSON', 'EPIPHONE', 'IBANEZ', 'PRS', 'ESP', 'SCHECTER',
       'JACKSON', 'SQUIER', 'GRETSCH', 'PEAVEY', 'B.C. RICH', 'CHARVEL',
       'DEAN', 'YAMAHA', 'G&L', 'WASHBURN', 'RICKENBACKER']:
                    self.brand = 'OTHER'
            else: self.brand = None
                    
            # MODEL OF GUITAR - DUMP INTO TEXT DESCRIPTION FOR NLP
            # ALSO 1-HOT ENCODE THE EXISTENCE OF THIS VARIABLE
            self.model = self.__attrs.get('Model')
            if self.model:
                self.model = self.model.upper()
                if "LES PAUL" in self.model:
                    self.model = "LES PAUL"
                if "TELECASTER" in self.model:
                    self.model = "TELECASTER"
                if "STRATOCASTER" in self.model:
                    self.model = "STRATOCASTER"
                if "STRAT" in self.model:
                    self.model = "STRATOCASTER"
                if "TELE" in self.model:
                    self.model = "TELECASTER"
                if "SG" in self.model:
                    self.model = "SG"
                if "FLYING V" in self.model:
                    self.model = "FLYING V"
                if "SQUIER" in self.model:
                    self.model = "SQUIER"
                if "EXPLORER" in self.model:
                    self.model = "EXPLORER"
                if "335" in self.model:
                    self.model = "335"
                if "339" in self.model:
                    self.model = "339"
                if "MUSTANG" in self.model:
                    self.model = "MUSTANG"
                if "JAGUAR" in self.model:
                    self.model = "JAGUAR"
#                 if self.model not in ['LES PAUL', 'STRATOCASTER', 'TELECASTER', 'SG', 'CUSTOM', 'VINTAGE',
#        '335', 'FLYING V', 'EXPLORER', 'CLASSIC', '339', 'JAGUAR', 'MUSTANG',
#        'PLUS', 'SINGLECUT', 'EC-1000', 'LEGACY', 'MELODY MAKER',
#        'ASAT CLASSIC', 'G-400', 'WOLFGANG USA', 'CUSTOM 24', 'CUSTOM 22',
#        'SE CUSTOM 24', 'STANDARD', 'CE 24', 'WOLFGANG SPECIAL', 'ES-175', '24',
#        'SQUIER']:
#                     self.model = "OTHER"
            
    
            # YEAR OF MANUFACTURE - IMPORTANT BINNABLE SCALAR WITH MANY MISSING VALUES 
            # 1-HOT ENCODE THE EXISTENCE OF THIS VAR
            if self.__attrs.get('Model Year'):
                self.year = self.__attrs.get('Model Year')[:4]
                if self.year:
                    try:
                        self.year = int(self.year)
                        if self.year == 86:
                            self.year = 1986
                        if self.year < 1700:
                            self.year = None
                    except ValueError:
                        self.year = None      
            else: 
                self.year = None
                
                
            # MATERIAL OF GUITAR BODY - 
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
                    
            # HANDEDNESS OF GUITAR (R/L) - 1-HOT THIS
            self.right_left_handed = self.__attrs.get('Right-/ Left-Handed')
            if self.right_left_handed:
                if "R" in self.right_left_handed.upper():
                    self.right_left_handed = "RIGHT"
                else:
                    self.right_left_handed = "LEFT"
            
            
            # COUNTRY OF MANUFACTURE - 1-HOT THIS
            self.country_manufacture = self.__attrs.get('Country/Region of Manufacture')
            if self.country_manufacture:
                self.country_manufacture = self.country_manufacture.upper()
                if self.country_manufacture == 'KOREA, REPUBLIC OF':
                    self.country_manufacture = 'KOREA'
                if self.country_manufacture == 'UNITED STATES':
                    self.country_manufacture = 'USA'
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
            
            # STRING CONFIG - 1-HOT CATEGORICAL
            self.string_config = self.__attrs.get('String Configuration')
            if self.string_config:
                self.string_config = self.string_config.upper()
                self.string_config = self.string_config.split()[0]
                if self.string_config.startswith("6"):
                    self.string_config = "6"
                if self.string_config.startswith("12"):
                    self.string_config = "12"
                if self.string_config == "SIX":
                    self.string_config = "6"
                if self.string_config == "TWELVE":
                    self.string_config = "12"
                try:
                    self.string_config = int(self.string_config)
                except TypeError:
                    self.string_config = None
                
                
            
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
            self.color = self.brand = self.model = None
            self.material = self.right_left_handed = self.country_manufacture = None
            self.body_type = self.string_config = self.listing_type = None    
            self.year = None
        
        # ATTEMPT TO SCRAPE YEAR FROM TEXT FIELDS
        year_pat = re.compile(r'(?<!NO )(?<!IN )(?<!MODEL )(?<!SINCE )(?<![\dA-Za-z-])(20[01]\d)(?!\d|\'|s)|(?<!NO )(?<!IN )(?<!MODEL )(?<!SINCE )(19[23456789]\d)(?!\d|\'|s)')
        
        if self.year == None or self.year == None:
            if re.findall(year_pat, self.title):
                self.year = re.findall(year_pat, self.title)[-1]
            elif self.subtitle and re.findall(year_pat, self.subtitle):
                self.year = re.findall(year_pat, self.subtitle)[-1]
            elif self.condition_description and re.findall(year_pat, self.condition_description):
                self.year = re.findall(year_pat, self.condition_description)[-1]
            elif self.description and re.findall(year_pat, self.description):
                self.year = re.findall(year_pat, self.description)[-1]
        if type(self.year) == tuple:
            try:
                self.year = int(self.year[0])
            except TypeError:
                self.year = int(self.year[1])