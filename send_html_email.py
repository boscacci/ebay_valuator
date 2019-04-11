from scrape_current_auctions import *

print(stars)
print('Formatting an email:')

email = "<h3>5 Most Undervalued:</h3>\n"

email += "*******************************\n"

for i in range(5):
    email += f"<a href = {m_u['Link'].values[i]}>"
    email += f"\n{m_u['Title'].values[i]}\n\n:"
    email += f"<img src = {m_u['Pic'].values[i]}></img></a>\n"
    email += f"\nCurrent Price: ${m_u['Price'].values[i]}"
    email += f"\nBids: {m_u['Bids'].values[i]}"
    email += f"\nEnding in: {m_u['Hours_til_close'].values[i]} Hours\n\n"
    email += "*******************************\n"
    
email += '\n<h3>5 Highest Potential Value:</h3>\n'
email += "*******************************\n"
    
for i in range(5):
    email += f"<a href = {hv_10['Link'].values[i]}>"
    email += f"\n{hv_10['Title'].values[i]}\n\n"
    email += f"<img src = {hv_10.Pic.values[i]}></img></a>\n"
    email += f"\nCurrent Price: ${hv_10['Price'].values[i]}"
    email += f"\nBids: {hv_10['Bids'].values[i]}"
    email += f"\nEnding in: {hv_10['Hours_til_close'].values[i]} Hours\n\n"
    email += "*******************************\n"

yag = yagmail.SMTP("gu1tarb1trag3@gmail.com")

address = input("Enter recipient email address: ")

yag.send(
    to=address,
    subject=f"eBay TrawlBot - Auctions Ending Within {hours_ahead} Hour(s)",
    contents=email)

print(f"Summary sent to {address.split('@')[0]}. Happy hunting")