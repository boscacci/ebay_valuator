# GuitArbitrage

## Trawls eBay for undervalued used guitars.

The idea was initially to train a model to estimate what the final auction price of your guitar would be - before you post it. Estimates would be based on the item sale settings: Auction duration, return policy, and other technical features, as well as the text you the seller type into the title, the description, and etc.

So I pulled a lot of real eBay data: about 15k closed, successful* used guitar auctions. I parsed and visualized the data, organized some seemingly relevant features, and threw all at a lasso regression model. Performance wasn't very good; RMSE was only about 18-25% better than just guessing the mean.

What I then realized was that it was a pretty conservative estimator, all said and done: If I flipped it into a classification problem ("will a guitar sell for more than X dollars?"), then I had way more false negatives than false positives. This led me to believe it wasn't totally useless.