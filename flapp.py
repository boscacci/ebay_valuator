from datetime import datetime
from flask import Flask
app = Flask(__name__)

from scrape_current_auctions import *

print(stars)
print('Formatting an html:')

html = f"Last updated {datetime.now()}\n"

html += "<h3>5 Most Undervalued:</h3>\n"

html += "*******************************\n"

for i in range(5):
    html += f"<a href = {m_u['Link'].values[i]}>"
    html += f"\n{m_u['Title'].values[i]}\n\n:"
    html += f"<img src = {m_u['Pic'].values[i]}></img></a>\n"
    html += f"\nCurrent Price: ${m_u['Price'].values[i]}"
    html += f"\nBids: {m_u['Bids'].values[i]}"
    html += f"\nEnding in: {m_u['Hours_til_close'].values[i]} Hours\n\n"
    html += "*******************************\n"
    
html += '\n<h3>5 Highest Potential Value:</h3>\n'
html += "*******************************\n"
    
for i in range(5):
    html += f"<a href = {hv_10['Link'].values[i]}>"
    html += f"\n{hv_10['Title'].values[i]}\n\n"
    html += f"<img src = {hv_10.Pic.values[i]}></img></a>\n"
    html += f"\nCurrent Price: ${hv_10['Price'].values[i]}"
    html += f"\nBids: {hv_10['Bids'].values[i]}"
    html += f"\nEnding in: {hv_10['Hours_til_close'].values[i]} Hours\n\n"
    html += "*******************************\n"

@app.route('/')
def shades():
    return html

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)