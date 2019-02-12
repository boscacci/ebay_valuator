# GuitArbitrage

Trawls eBay and notifies the user of undervalued used electric guitar auctions ending soon.

![](/_media/mailer.png)

## About

Estimates of value are based on: 
* __Item Characteristics__: Body material, country of manufacture, brand, color, right-or-left handed, etc. 
* __Item sale settings__: Auction duration, return policy, shipping fees, etc.
* __Other text input data__: Auction title, description, and other fields such as condition description and subtitle.

![](/_media/kepler_vis.gif)


![](/_media/object_init.png)

## Setup

__Prereq:__ A valid eBay API key saved to a file named "key"

Running the 3 python scripts in order sets you up for GuitArbitrage:

__00_persist_eBay_data__ : Populates a "data/axe_listings" directory with historical used electric guitar auctions as JSONs, with which to train a model. The more you can get, the better.

__01_train_regressor__ : Parses out the fetched JSONs and attempts to train a baseline Lasso regression model with whatever data you've made locally available. This includes TF-IDF transformed text bigrams as a feature, which may result in user-entered features like "hardshell_case" and "mint_condition" helping to determine price. This also contains commented-out code to train a TPOT pipeline if you want a more accurate / advanced model, and you also have a lot of time.

![](/_media/guitar_projector_small.gif)

__02_guitarbitrage__ : Fetches open eBay auctions and, using the trained model, tabulates a ratio between current high bids and predicted final sale prices. Formats an email for the user with 10 of the most "undervalued" guitars on the market closing soon.

Just trying to predict final sale price as a regression problem, predictive performance on a holdout test set wasn't terrific: RMSE was only about 27% better than just guessing the mean. However, as a classifier (trying to predict whether or not a guitar will sell above say $600) the model skews towards the right kind of error—lots of false negatives and not so many false positives, like less than 11% false positives with a robust model.

