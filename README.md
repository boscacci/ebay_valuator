# GuitArbitrage

## Trawls eBay for undervalued used guitars.

Estimates of value are based on: 
* __Item Characteristics__: Body material, country of manufacture, brand, color, right-or-left handed, etc. 
* __Item sale settings__: Auction duration, return policy, shipping fees, etc.
* __Other text input data__: Auction title, description, and other fields such as condition description and subtitle.

So I pulled a lot of real eBay data: about 15k closed used guitar auctions that ended in sales (caveat). I parsed and visualized the data, wrangled the seemingly relevant features, and threw them at a lasso regression model. This included TF-IDF transformed text bigrams, which resulted in features like "hardshell_case" and "mint_condition". 

Predictive performance on a holdout test set wasn't very good; RMSE was only about 18-25% better than just guessing the mean.

What I then realized was that mine was a pretty conservative estimator, price-wise: It usually guessed too low. If I re-framed it as a classification problem ("will a guitar sell for more than X dollars?"), then I had far more false negatives than false positives. This led me to believe it wasn't totally useless.

So now it's trained to trawl eBay for what it thinks are undervalued guitars, put together a little email with its findings, and send them wherever you like.

Grab an eBay API key and run the .py modules in order from 01 to 03 to generate your very own inbox clutter.

![](/_media/guitar_projector_small.gif)