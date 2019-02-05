# GuitArbitrage

## Trawls eBay for undervalued used guitars.

The idea was initially to train a model to estimate what the final auction price of your guitar would be - before you post it. 

Estimates would be based on the item sale settings and item characteristics: Auction duration, return policy, and other technical features, combined with the features you enter about the guitar (body material, country of manufacture, brand, color, right-or-left handed..etc), as well as the text you the seller type into the title, the description, and etc.

So I pulled a lot of real eBay data: about 15k closed used guitar auctions that ended in sales (caveat). I parsed and visualized the data, organized some seemingly relevant features, and threw all at a lasso regression model. Predictive performance on a holdout test set wasn't very good; RMSE was only about 18-25% better than just guessing the mean.

What I then realized was that it was a pretty conservative estimator, all said and done: It usually guessed too low. If I flipped it into a classification problem ("will a guitar sell for more than X dollars?"), then I had way more false negatives than false positives. This led me to believe it wasn't totally useless.

Grab an eBay API key and run the .py modules in order from 01 to 03 to generate your own inbox clutter.
