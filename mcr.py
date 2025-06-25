import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from textblob import TextBlob
import matplotlib.pyplot as plt
import re
import pandas as pd

# Configurações de estilo
FONT_COLOR = '#333333'  # Cor do texto
BAR_COLOR = '#000000'   # Preto para as barras
BACKGROUND_COLOR = '#e0e0e0'  # Cinza claro

# Dados
ALBUM_URL = "https://genius.com/albums/My-chemical-romance/The-black-parade"
STOPWORDS = {"the", "and", "to", "of", "i", "you", "this", "that", "all", "me", "a", "lyrics"}

# Processamento
response = requests.get(ALBUM_URL)
soup = BeautifulSoup(response.text, 'html.parser')
song_data = []
all_lyrics = ""

for link in [a['href'] for a in soup.find_all('a', href=True) 
            if '/My-chemical-romance-' in a['href'] and '-lyrics' in a['href']][:14]:
    try:
        page = requests.get(link)
        soup = BeautifulSoup(page.text, 'html.parser')
        title = soup.find('h1').text.replace('Lyrics', '').strip()
        
        lyrics_div = soup.find('div', {'data-lyrics-container': 'true'}) or soup.find('div', class_='lyrics')
        if lyrics_div:
            text = re.sub(r'\[.*?\]', '', lyrics_div.get_text())
            clean_text = ' '.join(text.split()).lower()
            all_lyrics += " " + clean_text
            
            analysis = TextBlob(text)
            song_data.append({
                'song': title,
                'polarity': analysis.sentiment.polarity
            })
            
    except Exception as e:
        print(f"Erro em {link}: {e}")

# Wordcloud
words = [word for word in re.findall(r'\b\w{4,}\b', all_lyrics) if word not in STOPWORDS]
wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color=BACKGROUND_COLOR,
    colormap="binary",
    stopwords=STOPWORDS
).generate(" ".join(words))

plt.figure(figsize=(15, 7), facecolor=BACKGROUND_COLOR)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout()
plt.savefig("mcr_wordcloud.png", dpi=300, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
plt.close()

# Gráfico de Sentimentos (horizontal)
df = pd.DataFrame(song_data).sort_values('polarity', ascending=True)

plt.figure(figsize=(15, 7), facecolor=BACKGROUND_COLOR)
ax = plt.gca()
ax.set_facecolor(BACKGROUND_COLOR)

# Barras horizontais
bars = plt.barh(
    df['song'], 
    df['polarity'], 
    color=BAR_COLOR,
    alpha=0.8,
    height=0.6
)

# Estilo do gráfico
plt.title('ANÁLISE DE SENTIMENTO - THE BLACK PARADE', 
        fontsize=18, pad=20, color=FONT_COLOR, fontweight='bold')
plt.xlabel('Polaridade Sentimental (-1 = Triste, +1 = Feliz)', 
        fontsize=12, color=FONT_COLOR)
plt.xticks(color=FONT_COLOR)
plt.yticks(color=FONT_COLOR, fontsize=10)

# Linha vertical no zero
plt.axvline(0, color='#666666', linestyle='--', linewidth=1)

# Remover bordas
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig("mcr_sentiment.png", dpi=300, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
plt.show()