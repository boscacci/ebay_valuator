
# Flatiron Data Science Immersive - Final Project Proposal


__What question/questions are you trying to solve?__
* How well can we guess the final auction price of an (X) on eBay, before we list it?
* How can I help my information security specialist friend classify and visualize incoming requests?


__What are the outcomes you think you will find, and then so what?__

* It is either trivial, challenging, or near-impossible to train an ML model to accurately guess the price of an (X)
* The cyber-security field is either rife with opportunity for data science innovation, or too far out of my domain expertise for me to be helpful


__How would someone, or some entity take action upon learning this?__
* In the event that my eBay model is confidently predictive, one could use it to help arbitrage goods across platforms.
* In the event that my info-sec tool is useful, white-hat pen-testers could use it to visualize apache logs and things.


__What version this question would allow me to find an answer in 2-3 days?__
* How accurate is an eBay predictive model that just goes off of the basic features (with no NLP or CNN action)?
* What if we just visualized incoming web requests / Fail2ban logs?


__What version of this question would allow me/motivate me to work on this problem even after completing Flatiron School?__
* For which eBay product category can I make the most accurate valuator model? Can I put it into practice?
* What other tools can I make for the information security field?


__What are some data sources that would allow you to answer this?__
* eBay API
* My friend in the NAVY


__What is the ideal data you would hope to gather to answer this question?__
* Tens of thousands of previous eBay auctions
* Tens of thousands of server logs and things


__What about potentially missing data, that could cause omitted variable bias?__
* There might be a lot of this, so I will have to get creative about making my sparse data less sparse
* Server logs will not have this problem mostly.


__How will you present your data? Will it be a jupyter notebook presentation or a Dashboard.__
* Dash was a good time, would use again
* Ideally users can make a new eBay listing, run it through the model to get a price guess
* Server log visualizations would be a geo-viz of some kind like a kepler.gl object of some kind


__How will you use regression and/or classification?__
* eBay value estimator will obviously guess the price of new listings
* Info-Sec project would probably try to classify incoming requests as malicious, safe, or unknown or something


__What are the challenges you foresee with this project?__
* Figuring out good model stacking and ninja feature engineering techniques
* Domain-specific challenges having to do with the eBay API responses
* 5000-request daily limit with eBay API


__What are your next steps moving forward?__
* Making hecka requests to the eBay API and collecting data
* Poking and prodding the data every which way possible
* NLP pipeline for titles, description, condition description text
* Think about how I might use the images to help guess price
