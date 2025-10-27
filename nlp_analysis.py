# NLP Analysis with spaCy - Named Entity Recognition & Sentiment
import spacy
from spacy import displacy
import matplotlib.pyplot as plt

print("=== NLP Analysis with spaCy ===")

# Load English model
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded successfully")
except OSError:
    print("❌ Please download the model first: python -m spacy download en_core_web_sm")
    exit()

# Sample product reviews dataset
reviews = [
    "I love my new Kindle Paperwhite from Amazon. The battery life is incredible and the display is perfect for reading.",
    "This Hoover vacuum cleaner is terrible. It broke after two weeks of use and the customer service was awful.",
    "I bought Sony WH-1000XM4 headphones and they have amazing noise cancellation. Best purchase ever!",
    "The Samsung Galaxy S23 has a great camera but the battery life could be better.",
    "Do not buy this Dyson fan. It stopped working after one month and the company refused to refund my money.",
    "Apple MacBook Pro with M2 chip is fantastic for programming and video editing.",
    "This Lenovo laptop constantly crashes and the build quality is poor for the price."
]

print("Performing NER and Sentiment Analysis...\n")

results = []

for i, review in enumerate(reviews, 1):
    doc = nlp(review)
    
    # Extract entities (products and brands)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]]
    
    # Rule-based sentiment analysis
    positive_words = {"love", "incredible", "amazing", "great", "best", "perfect", "excellent", "good", "fantastic"}
    negative_words = {"terrible", "broke", "awful", "bad", "worst", "refused", "stopped", "poor", "crashes"}
    
    positive_count = sum(1 for token in doc if token.lemma_.lower() in positive_words)
    negative_count = sum(1 for token in doc if token.lemma_.lower() in negative_words)
    
    if positive_count > negative_count:
        sentiment = "POSITIVE"
    elif negative_count > positive_count:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    
    results.append({
        'review': review,
        'entities': entities,
        'sentiment': sentiment,
        'positive_count': positive_count,
        'negative_count': negative_count
    })
    
    print(f"Review {i}:")
    print(f"Text: {review}")
    print(f"Entities: {entities}")
    print(f"Sentiment: {sentiment} (+{positive_count}/-{negative_count})")
    print("-" * 80)

# Visualization
sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
for result in results:
    sentiment_counts[result['sentiment']] += 1

# Sentiment distribution pie chart
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
colors = ['#4CAF50', '#F44336', '#FFC107']
plt.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', colors=colors)
plt.title('Sentiment Distribution in Reviews')

# Entity extraction count
all_entities = []
for result in results:
    all_entities.extend([ent[0] for ent in result['entities']])

from collections import Counter
entity_counts = Counter(all_entities)

plt.subplot(1, 2, 2)
if entity_counts:
    entities, counts = zip(*entity_counts.most_common(5))
    plt.bar(entities, counts, color='skyblue')
    plt.title('Top 5 Extracted Entities')
    plt.xticks(rotation=45)
else:
    plt.text(0.5, 0.5, 'No entities extracted', ha='center', va='center')
    plt.title('Entity Extraction Results')

plt.tight_layout()
plt.savefig('nlp_analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== SUMMARY ===")
print(f"Total reviews analyzed: {len(reviews)}")
print(f"Positive reviews: {sentiment_counts['POSITIVE']}")
print(f"Negative reviews: {sentiment_counts['NEGATIVE']}")
print(f"Neutral reviews: {sentiment_counts['NEUTRAL']}")
print(f"Total entities extracted: {len(all_entities)}")
