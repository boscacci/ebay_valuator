import json, re
from datetime import datetime

class Shade_Sale_Memory:
    def __init__(self, listing, specs):
        
        # UID
        self.__body = {'listing': listing, 'specs': specs}
        self.id = self.__body['listing']['itemId'][0]
        
        
        # ULTIMATE PRICE - SCALAR DEPENDENT VAR
        if self.__body['specs'].get('ConvertedCurrentPrice'):
            self.price = self.__body['specs']['ConvertedCurrentPrice']['Value']
        else: self.price = None

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
        if self.listing_type == "AuctionWithBIN":
            self.listing_type = "AUCTION"
        self.listing_type = self.listing_type.upper()
        if not self.listing_type:
            self.listing_type = "UNLISTED"
        
        ################# SHIPPING STUFF #################
        
        # SHIPPING TYPE - 1-HOT CATEGORICAL
        self.ship_type = self.__body['listing']['shippingInfo'][0]['shippingType'][0]
        self.ship_type = self.ship_type.upper()
        if self.ship_type not in ['CALCULATED', 'FLAT', 'FREE']:
            self.ship_type = 'OTHER'
        
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
            self.seller_feedback_score = 0
        
        # BEST OFFER ENABLED - ONE-HOT ENCODE
        self.best_offer_enabled = self.__body['specs']['BestOfferEnabled']
        
        ################# NITTY GRITTIES #################
        
        # GET THE ITEM SPECIFICS FOR FURTHER PARSING
        if self.__body['specs'].get('ItemSpecifics'):
            self.__attrs = {prop['Name']:prop['Value'][0] 
                            for prop in self.__body['specs']['ItemSpecifics']['NameValueList']}
            
            # BRAND / MAKE OF ITEM - 1-HOT ENCODE
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
            

            # MODEL OF ITEM - 1-HOT ENCODE
            # TOSS INTO NLP STEW

            self.model = self.__attrs.get('Model')
            if self.model:
                self.model = self.model.upper()
                if 'RAY BAN' in self.model or 'RAYBAN' in self.model:
                    self.model = 'RAY-BAN'
            else: self.model = "UNLISTED"

            # FRAME MATERIAL
            # TOSS IT INTO THE NLP STEW
            self.frame_material = self.__attrs.get('Frame Material')
            if self.frame_material:
                self.frame_material = self.frame_material.upper()
                if "PLASTIC" in self.frame_material and "METAL" not in self.frame_material:
                    self.frame_material = "PLASTIC"
                if "METAL" in self.frame_material and "PLASTIC" not in self.frame_material:
                    self.frame_material = "METAL"
                if "METAL" in self.frame_material and "PLASTIC" in self.frame_material:
                    self.frame_material = "METAL_AND_PLASTIC"
                if "NYLON" in self.frame_material and "PLASTIC" not in self.frame_material\
                and "METAL" not in self.frame_material:
                    self.frame_material = "NYLON"
                if "OMATTER" in self.frame_material or "O-MATTER" in self.frame_material or\
                "O MATTER" in self.frame_material:
                    self.frame_material = "O-MATTER"
                if "GOLD" in self.frame_material:
                    self.frame_material = "GOLD"
                if "GRILAMID" in self.frame_material:
                    self.frame_material = "GRILAMID"
                if "WOOD" in self.frame_material:
                    self.frame_material = "WOOD"
                if self.frame_material not in ['PLASTIC', 'METAL', 'METAL_AND_PLASTIC']:
                    self.frame_material = "OTHER"
            else: self.frame_material = "UNLISTED"

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
            else: self.country_manufacture = "UNLISTED"
            
            
            # LENS TECH - 1-HOT CATEGORICAL
            self.lens_tech = self.__attrs.get('Lens Technology')
            if self.lens_tech:
                self.lens_tech = self.lens_tech.upper()
                if "POLARIZED" and "MIRROR" in self.lens_tech:
                    self.lens_tech = "POLARIZED_MIRRORED"
                elif "GRADIENT" and "MIRROR" in self.lens_tech:
                    self.lens_tech = "GRADIENT_MIRRORED"
                elif "IRIDIUM" in self.lens_tech:
                    self.lens_tech = "IRIDIUM"
                elif "G" in self.lens_tech and "15" in self.lens_tech:
                    self.lens_tech = "G15"
                elif "POLARIZED" in self.lens_tech or "POLARIZD" in self.lens_tech:
                    if "NON" not in self.lens_tech and "NOT" not in self.lens_tech:
                        self.lens_tech = "POLARIZED"
                if self.lens_tech not in ['POLARIZED', 'POLARIZED_MIRRORED', 'ANTI-REFLECTIVE', 'GRADIENT',
                                            'UNLISTED', 'IRIDIUM']:
                    self.lens_tech = "OTHER"
            else: self.lens_tech = "UNLISTED"
            
            # FRAME COLOR - DUMP (+ "COLORED" for bigram) INTO NLP WITH DESCRIPTION
            # ONE-HOT ENCODE THE EXISTENCE OF THIS VARIABLE
            self.frame_color = self.__attrs.get('Frame Color')
            if self.frame_color:
                self.frame_color = self.frame_color.upper()
                if "CHERRY" in self.frame_color:
                    self.frame_color = "RED"
                if "BURST" in self.frame_color:
                    self.frame_color = "SUNBURST"
                if "MAHOGANY" in self.frame_color:
                    self.frame_color = "BROWN"
                if "BLUE" in self.frame_color:
                    self.frame_color = "BLUE"
                if "TURQUOISE" in self.frame_color:
                    self.frame_color = "BLUE"
                if "TEAL" in self.frame_color:
                    self.frame_color = "BLUE"
                if "RED" in self.frame_color:
                    self.frame_color = "RED"
                if "BLACK" in self.frame_color:
                    self.frame_color = "BLACK"
                if "EBONY" in self.frame_color:
                    self.frame_color = "BLACK"
                if "WHITE" in self.frame_color:
                    self.frame_color = "WHITE"
                if "GREEN" in self.frame_color:
                    self.frame_color = "GREEN"
                if "NATURAL" in self.frame_color:
                    self.frame_color = "BROWN"
                if "BLONDE" in self.frame_color:
                    self.frame_color = "BROWN"
                if "BLOND" in self.frame_color:
                    self.frame_color = "BROWN"
                if "BEIGE" in self.frame_color:
                    self.frame_color = "BROWN"
                if "MAPLE" in self.frame_color:
                    self.frame_color = "BROWN"
                if "BUTTERSCOTCH" in self.frame_color:
                    self.frame_color = "BROWN"
                if "WALNUT" in self.frame_color:
                    self.frame_color = "BROWN"
                if "TOBACCO" in self.frame_color:
                    self.frame_color = "BROWN"
                if "BROWN" in self.frame_color:
                    self.frame_color = "BROWN"
                if "CREAM" in self.frame_color:
                    self.frame_color = "WHITE"
                if "YELLOW" in self.frame_color:
                    self.frame_color = "YELLOW"
                if "FIREGLO" in self.frame_color:
                    self.frame_color = "RED"
                if "WINE" in self.frame_color:
                    self.frame_color = "RED"
                if "BURGANDY" in self.frame_color:
                    self.frame_color = "RED"
                if "BURGUNDY" in self.frame_color:
                    self.frame_color = "RED"
                if "MULTI-COLOR" in self.frame_color:
                    self.frame_color = "MULTICOLOR"
                if "AMBER" in self.frame_color:
                    self.frame_color = "YELLOW"
                if "WOOD" in self.frame_color:
                    self.frame_color = "BROWN"
                if "COPPER" in self.frame_color:
                    self.frame_color = "RED"
                if "PEWTER" in self.frame_color:
                    self.frame_color = "GRAY"
                if "GRAY" in self.frame_color:
                    self.frame_color = "GRAY"
                if "TORTOISE" in self.frame_color:
                    self.frame_color = "TORTOISE"
                if self.frame_color not in ['BLACK', 'UNLISTED', 'GOLD', 'BROWN', 'SILVER', 'GRAY', 'WHITE']:
                    self.frame_color = "OTHER"
            else: self.frame_color = "UNLISTED"

            # LENS COLOR
            self.lens_color = self.__attrs.get('Lens Color')
            if self.lens_color:
                self.lens_color = self.lens_color.upper()
                if "CHERRY" in self.lens_color:
                    self.lens_color = "RED"
                if "BURST" in self.lens_color:
                    self.lens_color = "SUNBURST"
                if "MAHOGANY" in self.lens_color:
                    self.lens_color = "BROWN"
                if "BLUE" in self.lens_color:
                    self.lens_color = "BLUE"
                if "TURQUOISE" in self.lens_color:
                    self.lens_color = "BLUE"
                if "TEAL" in self.lens_color:
                    self.lens_color = "BLUE"
                if "RED" in self.lens_color:
                    self.lens_color = "RED"
                if "BLACK" in self.lens_color:
                    self.lens_color = "BLACK"
                if "EBONY" in self.lens_color:
                    self.lens_color = "BLACK"
                if "WHITE" in self.lens_color:
                    self.lens_color = "WHITE"
                if "GREEN" in self.lens_color:
                    self.lens_color = "GREEN"
                if "NATURAL" in self.lens_color:
                    self.lens_color = "BROWN"
                if "BLONDE" in self.lens_color:
                    self.lens_color = "BROWN"
                if "BLOND" in self.lens_color:
                    self.lens_color = "BROWN"
                if "BEIGE" in self.lens_color:
                    self.lens_color = "BROWN"
                if "MAPLE" in self.lens_color:
                    self.lens_color = "BROWN"
                if "BUTTERSCOTCH" in self.lens_color:
                    self.lens_color = "BROWN"
                if "WALNUT" in self.lens_color:
                    self.lens_color = "BROWN"
                if "TOBACCO" in self.lens_color:
                    self.lens_color = "BROWN"
                if "BROWN" in self.lens_color:
                    self.lens_color = "BROWN"
                if "CREAM" in self.lens_color:
                    self.lens_color = "WHITE"
                if "YELLOW" in self.lens_color:
                    self.lens_color = "YELLOW"
                if "FIREGLO" in self.lens_color:
                    self.lens_color = "RED"
                if "WINE" in self.lens_color:
                    self.lens_color = "RED"
                if "BURGANDY" in self.lens_color:
                    self.lens_color = "RED"
                if "BURGUNDY" in self.lens_color:
                    self.lens_color = "RED"
                if "MULTI-COLOR" in self.lens_color:
                    self.lens_color = "MULTICOLOR"
                if "AMBER" in self.lens_color:
                    self.lens_color = "YELLOW"
                if "WOOD" in self.lens_color:
                    self.lens_color = "BROWN"
                if "COPPER" in self.lens_color:
                    self.lens_color = "RED"
                if "PEWTER" in self.lens_color:
                    self.lens_color = "GRAY"
                if "GRAY" in self.lens_color:
                    self.lens_color = "GRAY"
                if "TORTOISE" in self.lens_color:
                    self.lens_color = "TORTOISE"
                if self.lens_color not in ['BLACK', 'UNLISTED', 'GOLD', 'BROWN', 'SILVER', 'GRAY', 'WHITE']:
                    self.lens_color = "OTHER"
            else: self.lens_color = "UNLISTED"

            # TEMPLE LENGTH
            self.temple_length_binary = self.__attrs.get('Temple Length') != None

            # STYLE
            self.style = self.__attrs.get('Style')
            if self.style:
                self.style = self.style.upper()
                if "AVIATOR" in self.style:
                        self.style = "AVIATOR"
                if "ATHLETIC" in self.style:
                        self.style = "SPORT"
                if "SPORT" in self.style:
                        self.style = "SPORT"
                if "SQUARE" in self.style:
                        self.style = "SQUARE"
                if "RECTANG" in self.style:
                        self.style = "RECTANGULAR"
                if self.style not in ['SPORT', 'VINTAGE', 'PILOT', 'RECTANGULAR', 'WRAP', 'SQUARE',
                                      'DESIGNER', 'AVIATOR']:
                    self.style = "OTHER"
            else: self.style = "UNLISTED"

            # PROTECTION
            self.protection = self.__attrs.get('Protection')
            if self.protection:
                self.protection = self.protection.upper()
                if self.protection not in ['100% UVA & UVB', '100% UV', '100% UV400', 'UNLISTED']:
                    self.protection = "OTHER"
            else: self.protection = "UNLISTED"
        
        # INITIALIZING VARIABLES THAT DIDN'T GET ASSIGNED VALUES
        else:
            self.frame_color = self.brand = self.lens_tech = 'UNLISTED'
            self.frame_material = self.country_manufacture = 'UNLISTED'
            self.lens_color = self.style = 'UNLISTED'
            self.protection = self.model = "UNLISTED"
            self.temple_length_binary = False

        # Reminder to self, put model name in NLP stew
