# main.py

import uvicorn
import os
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = FastAPI()

# Load the model
model_rating = tf.keras.models.load_model('TrackMate_Model.h5')  # Sesuaikan dengan path yang benar

# Load the scaler
scaler = StandardScaler()

# Load the dataset
df = pd.read_csv('TrackMate Dataset.csv')  # Sesuaikan dengan path yang benar
df_new = df.copy()

# Add new data
data_produk_baru_1 = [
    {
    'product_id': 'B112X5ADO8',
    'product_name': 'Greenbean Coffee',
    'category': 'Grocery and Gourmet Food',
    'discounted_price': '₹80,000',
    'actual_price': '₹80,000',
    'discount_percentage': '0%',
    'rating': 4.9,
    'rating_count': '7',
    'about_product': 'Cerry kopi yang sudah mengalami proses pasca panen dan siap panggang',
    'user_id': "AGSGSRTEZBQY64WO2HKQTV7TWFSA,AEYD5HVYAJ23CR6PTWOOIKUOIDHA,AFRMNW6TDHDZBP2UHF2K3MEAEYUA,AHICHCW6EC3BNV2IDAEAJPBG4HZQ,AGWFKE7RNP6EVC4JFLFSL76EEVVQ,AGEOQQHGNELZNEUKJAJUA7NTPBLA,AFS3QBSOMCE2FAZFUYZ3NBFQDLMQ,AGJYG6ZWCWD74WNE6Y37XZ2VUSMA",
    'user_name': "Birendra ku Dash,Aditya Gupta,Abdulla A N,Deepak,Gowtham,Rakesh,Pawan Kumar,Prabhat Raj Pathak",
    'review_id': "A1B2C3D4E5F6G7H8, I9J1K2L3M4N5O6, P7Q8R9S1T2U3V4, W5X6Y7Z8A9B1C2, D3E4F5G6H7I8J9, K1L2M3N4O5P6Q7, R8S9T1U2V3W4X5, Y6Z7A8B9C1D2E3",
    'review_title': "Rasa autentik dan khas, bahan-bahan segar terasa, penyajian cantik dan instagramable, menu ramah vegetarian, tempat yang bersih dan nyaman, suasana tenang, pilihan sehat dan bergizi, layanan ramah.",
    'review_content': 'Revolutionary, high-quality, elegant design, affordable, user-friendly, functional, fast delivery, highly recommended for any product!',
    'img_link': 'https://drive.google.com/open?id=1mCbKdwvUu9lZV-uC18Ak9_oAACyy18YH',
    'product_link': 'https://www.instagram.com/agroniaga_idn/'
},
    {
    'product_id': 'B613M6MSA8',
    'product_name': 'MORINGA TEA',
    'category': 'Grocery and Gourmet Food',
    'discounted_price': '₹40,000',
    'actual_price': '₹40,000',
    'discount_percentage': '0%',
    'rating': 4.9,
    'rating_count': '7',
    'about_product': "Produk berbahan dasar daun kelor yang bentuk kemasan teabag tinggal seduh.",
    'user_id': "AGSGSRTEZBQY64WO2HKQTV7TWFSA,AEYD5HVYAJ23CR6PTWOOIKUOIDHA,AFRMNW6TDHDZBP2UHF2K3MEAEYUA,AHICHCW6EC3BNV2IDAEAJPBG4HZQ,AGWFKE7RNP6EVC4JFLFSL76EEVVQ,AGEOQQHGNELZNEUKJAJUA7NTPBLA,AFS3QBSOMCE2FAZFUYZ3NBFQDLMQ,AGJYG6ZWCWD74WNE6Y37XZ2VUSMA",
    'user_name': "Birendra ku Dash,Aditya Gupta,Abdulla A N,Deepak,Gowtham,Rakesh,Pawan Kumar,Prabhat Raj Pathak",
    'review_id': "AGSGSRTEZBQY64WO2HKQTV7TWFSA,AEYD5HVYAJ23CR6PTWOOIKUOIDHA,AFRMNW6TDHDZBP2UHF2K3MEAEYUA,AHICHCW6EC3BNV2IDAEAJPBG4HZQ,AGWFKE7RNP6EVC4JFLFSL76EEVVQ,AGEOQQHGNELZNEUKJAJUA7NTPBLA,AFS3QBSOMCE2FAZFUYZ3NBFQDLMQ,AGJYG6ZWCWD74WNE6Y37XZ2VUSMA",
    'review_title': "Makanan sehat dan organik, pilihan vegan yang baik, desain interior yang Instagramable, suasana yang tenang dan ramah lingkungan, menu salad yang kreatif, harga terjangkau untuk makanan sehat, pelayanan yang ramah dan informatif.",
    'review_content': 'Revolutionary, high-quality, elegant design, affordable, user-friendly, functional, fast delivery, highly recommended for any product!',
    'img_link': 'https://drive.google.com/open?id=18osvUd5nE_Uu21d6c3aqi7ccPbT6dxOQ',
    'product_link': 'https://www.tokopedia.com/archive-newedenmoringa-1645404339/moringa-tea-new-eden-20gr?extParam=src%3Dshop%26whid%3D9669096'
},
    {
         'product_id': 'B404F7FMA1',
    'product_name': 'TEMPEKITA TEMPE SEGAR',
    'category': 'Grocery and Gourmet Food',
    'discounted_price': '₹17,000',
    'actual_price': '₹17,000',
    'discount_percentage': '0%',
    'rating': 4.9,
    'rating_count': '7',
    'about_product': "Tempe segar adalah produk yang berasal dari kacang kedelai yang difermentasi menggunakan ragi tempe.",
    'user_id': "AGSGSRTEZBQY64WO2HKQTV7TWFSA,AEYD5HVYAJ23CR6PTWOOIKUOIDHA,AFRMNW6TDHDZBP2UHF2K3MEAEYUA,AHICHCW6EC3BNV2IDAEAJPBG4HZQ,AGWFKE7RNP6EVC4JFLFSL76EEVVQ,AGEOQQHGNELZNEUKJAJUA7NTPBLA,AFS3QBSOMCE2FAZFUYZ3NBFQDLMQ,AGJYG6ZWCWD74WNE6Y37XZ2VUSMA",
    'user_name': "Birendra ku Dash,Aditya Gupta,Abdulla A N,Deepak,Gowtham,Rakesh,Pawan Kumar,Prabhat Raj Pathak",
    'review_id': "M8N9O1P2Q3R4S5T6U7V8W9, X1Y2Z3A4B5C6D7E8F9G1H2I3J4K5, L6M7N8O9P1Q2R3S4T5U6V7W8X9, Y1Z2A3B4C5D6E7F8G9H1I2J3K4L5, M6N7O8P9Q1R2S3T4U5V6W7X8Y9, Z1A2B3C4D5E6F7G8H9I1J2K3L4, M5N6O7P8Q9R1S2T3U4V5W6, X7Y8Z9A1B2C3D4E5F6G7",
    'review_title': "Makanan sehat dan organik, pilihan vegan yang baik, desain interior yang Instagramable, suasana yang tenang dan ramah lingkungan, menu salad yang kreatif, harga terjangkau untuk makanan sehat, pelayanan yang ramah dan informatif.",
    'review_content': 'Revolutionary, high-quality, elegant design, affordable, user-friendly, functional, fast delivery, highly recommended for any product!',
    'img_link': 'https://drive.google.com/open?id=1fv6EFhTNGeAuZgMapiOODILB3um7LgOX',
    'product_link': 'https://shopee.co.id/-Tempekita.id-Tempe-Segar-i.419555041.8543489708?xptdk=8ec5c78a-ef7b-4600-b5ab-8652b22055eb'
    },
    {
        'product_id': 'B772O6JWR8',
    'product_name': 'ZAKET',
    'category': 'Home&Kitchen',
    'discounted_price': '₹28,000',
    'actual_price': '₹28,000',
    'discount_percentage': '0%',
    'rating': 4.9,
    'rating_count': '7',
    'about_product': "Terbuat dari limbah sawit",
    'user_id': "AGSGSRTEZBQY64WO2HKQTV7TWFSA,AEYD5HVYAJ23CR6PTWOOIKUOIDHA,AFRMNW6TDHDZBP2UHF2K3MEAEYUA,AHICHCW6EC3BNV2IDAEAJPBG4HZQ,AGWFKE7RNP6EVC4JFLFSL76EEVVQ,AGEOQQHGNELZNEUKJAJUA7NTPBLA,AFS3QBSOMCE2FAZFUYZ3NBFQDLMQ,AGJYG6ZWCWD74WNE6Y37XZ2VUSMA",
    'user_name': "Birendra ku Dash,Aditya Gupta,Abdulla A N,Deepak,Gowtham,Rakesh,Pawan Kumar,Prabhat Raj Pathak",
    'review_id': "O8P9Q1R2S3T4U5V6W7X8Y9, Z1A2B3C4D5E6F7G8H9I1J2K3, L4M5N6O7P8Q9R1S2T3U4V5W6, X7Y8Z9A1B2C3D4E5F6G7H8, I9J1K2L3M4N5O6P7Q8R9S1T2, U3V4W5X6Y7Z8A9B1C2D3E4F5, G6H7I8J9K1L2M3N4O5P6, Q7R8S9T1U2V3W4X5Y6Z7",
    'review_title': "Arang dari limbah ini adalah solusi ramah lingkungan yang brilian, Mengubah limbah menjadi sumber energi yang berguna, ini membantu mengurangi jejak karbon, Proses daur ulang limbah menjadi arang sangat inovatif dan membantu menjaga lingkungan, Saya sangat mendukung produk-produk yang memprioritaskan keberlanjutan, arang dari limbah ini adalah langkah positif menuju masa depan yang lebih hijau dan berkelanjutan.",
    'review_content': 'Revolutionary, high-quality, elegant design, affordable, user-friendly, functional, fast delivery, highly recommended for any product!',
    'img_link': 'https://drive.google.com/open?id=1JTMePUNF4XMWe8IW5VW5et513nESlgWB',
    'product_link': 'https://drive.google.com/open?id=1JTMePUNF4XMWe8IW5VW5et513nESlgWB'
    },
    {
         'product_id': 'B103G6MHZ9',
    'product_name': 'Mutiara Beras Glukomanan Porang',
    'category': 'Grocery and Gourmet Food',
    'discounted_price': '₹135,000',
    'actual_price': '₹135,000',
    'discount_percentage': '0%',
    'rating': 4.9,
    'rating_count': '7',
    'about_product': "Nasi tinggi serat rendah karbohidrat rendah kalori dan rendah gula",
    'user_id': "AGSGSRTEZBQY64WO2HKQTV7TWFSA,AEYD5HVYAJ23CR6PTWOOIKUOIDHA,AFRMNW6TDHDZBP2UHF2K3MEAEYUA,AHICHCW6EC3BNV2IDAEAJPBG4HZQ,AGWFKE7RNP6EVC4JFLFSL76EEVVQ,AGEOQQHGNELZNEUKJAJUA7NTPBLA,AFS3QBSOMCE2FAZFUYZ3NBFQDLMQ,AGJYG6ZWCWD74WNE6Y37XZ2VUSMA",
    'user_name': "Birendra ku Dash,Aditya Gupta,Abdulla A N,Deepak,Gowtham,Rakesh,Pawan Kumar,Prabhat Raj Pathak",
    'review_id': "A8B9C1D2E3F4G5H6I7J8, K9L1M2N3O4P5Q6R7S8T9U1V2W3, X4Y5Z6A7B8C9D1E2F3G4H5I6J7, K8L9M1N2O3P4Q5R6S7T8U9V1W2, X3Y4Z5A6B7C8D9E1F2G3H4I5, J6K7L8M9N1O2P3Q4R5S6T7, U8V9W1X2Y3Z4A5B6C7D8, E9F1G2H3I4J5K6L7M8N9O1P2",
    'review_title': "Makanan sehat dan organik, pilihan vegan yang baik, desain interior yang Instagramable, suasana yang tenang dan ramah lingkungan, menu salad yang kreatif, harga terjangkau untuk makanan sehat, pelayanan yang ramah dan informatif.",
    'review_content': 'Revolutionary, high-quality, elegant design, affordable, user-friendly, functional, fast delivery, highly recommended for any product!',
    'img_link': 'https://drive.google.com/open?id=1xUJLdhbxytbRK9u1IdTXN9h3jETHAJVi',
    'product_link': 'https://dapurporang.com/'
    },
      {   'product_id': 'B307H9TVZ8',
    'product_name': 'Nasi Jagung Instan Loyangku',
    'category': 'Grocery and Gourmet Food',
    'discounted_price': '₹15,000',
    'actual_price': '₹15,000',
    'discount_percentage': '0%',
    'rating': 4.9,
    'rating_count': '7',
    'about_product': "Nasi Jagung Instan Loyangku terbuat dari jagung pilihan dari petani jagung Banjarnegara yang diolah secara tradisioanal oleh masyarakat Desa Pucungbedug. Yuk hidup sehat bersama Nasi Jagung Instan Loyangku.",
    'user_id': "AGSGSRTEZBQY64WO2HKQTV7TWFSA,AEYD5HVYAJ23CR6PTWOOIKUOIDHA,AFRMNW6TDHDZBP2UHF2K3MEAEYUA,AHICHCW6EC3BNV2IDAEAJPBG4HZQ,AGWFKE7RNP6EVC4JFLFSL76EEVVQ,AGEOQQHGNELZNEUKJAJUA7NTPBLA,AFS3QBSOMCE2FAZFUYZ3NBFQDLMQ,AGJYG6ZWCWD74WNE6Y37XZ2VUSMA",
    'user_name': "Birendra ku Dash,Aditya Gupta,Abdulla A N,Deepak,Gowtham,Rakesh,Pawan Kumar,Prabhat Raj Pathak",
    'review_id': "Q3R4S5T6U7V8W9X1Y2Z3A4, B5C6D7E8F9G1H2I3J4K5L6M7N8O9, P1Q2R3S4T5U6V7W8X9Y1Z2A3B4C5D6E7, F8G9H1I2J3K4L5M6N7O8P9Q1R2S3T4, U5V6W7X8Y9Z1A2B3C4D5E6F7G8H9, I1J2K3L4M5N6O7P8Q9R1S2T3U4V5, W6X7Y8Z9A1B2C3D4E5F6G7H8, I9J1K2L3M4N5O6P7Q8R9S1T2",
    'review_title': "Makanan sehat dan organik, pilihan vegan yang baik, desain interior yang Instagramable, suasana yang tenang dan ramah lingkungan, menu salad yang kreatif, harga terjangkau untuk makanan sehat, pelayanan yang ramah dan informatif.",
    'review_content': 'Revolutionary, high-quality, elegant design, affordable, user-friendly, functional, fast delivery, highly recommended for any product!',
    'img_link': 'https://drive.google.com/open?id=1auE4DYUhpOjOSn-P_Fk7akmAHzNoCj7e',
    'product_link': 'https://gubug-eva.business.site/'
    },
     {    'product_id': 'B397P6QHS8',
    'product_name': 'Sambal Baby Cumi',
    'category': 'Grocery and Gourmet Food',
    'discounted_price': '₹40,000',
    'actual_price': '₹40,000',
    'discount_percentage': '0%',
    'rating': 4.9,
    'rating_count': '7',
    'about_product': 'Sambal yang terbuat dari baby cumi asin yang berasal dari daerah desa Gerokgak kec. Gerokgak Kab. Buleleng beserta aneka cabai dan rempah-rempah bumbu segar tanpa pengawet yang diolah dengan cara dimasak sehingga memiliki rasa yang pedas, gurih dan nikmat. Cocok jika dimakan bersama nasi panas.',
    'user_id': "AGSGSRTEZBQY64WO2HKQTV7TWFSA,AEYD5HVYAJ23CR6PTWOOIKUOIDHA,AFRMNW6TDHDZBP2UHF2K3MEAEYUA,AHICHCW6EC3BNV2IDAEAJPBG4HZQ,AGWFKE7RNP6EVC4JFLFSL76EEVVQ,AGEOQQHGNELZNEUKJAJUA7NTPBLA,AFS3QBSOMCE2FAZFUYZ3NBFQDLMQ,AGJYG6ZWCWD74WNE6Y37XZ2VUSMA",
    'user_name': "Birendra ku Dash,Aditya Gupta,Abdulla A N,Deepak,Gowtham,Rakesh,Pawan Kumar,Prabhat Raj Pathak",
    'review_id': "U3V4W5X6Y7Z8A9B1C2D3E4F5, G6H7I8J9K1L2M3N4O5P6Q7R8S9, T1U2V3W4X5Y6Z7A8B9C1D2E3F4G5, H6I7J8K9L1M2N3O4P5Q6R7S8T9U1, V2W3X4Y5Z6A7B8C9D1E2F3G4H5, I6J7K8L9M1N2O3P4,D3E4F5G6H7I8J9, K1L2M3N4O5P6Q7",
    'review_title': "Makanan sehat dan organik, pilihan vegan yang baik, desain interior yang Instagramable, suasana yang tenang dan ramah lingkungan, menu salad yang kreatif, harga terjangkau untuk makanan sehat, pelayanan yang ramah dan informatif.",
    'review_content': 'Revolutionary, high-quality, elegant design, affordable, user-friendly, functional, fast delivery, highly recommended for any product!',
    'img_link': 'https://www.instagram.com/sambal_mamo/',
    'product_link': 'https://shopee.co.id/sambal_mamo-i.997481767.21685181932?sp_atk=3e60a7ec-5e55-431a-9527-c1ee7d2dca32&xptdk=3e60a7ec-5e55-431a-9527-c1ee7d2dca32'
    }
    ]
df_new = df_new.append(data_produk_baru_1, ignore_index=True)
df_new.to_csv('TrackMate Dataset.csv', index=False)

# Drop missing value
df_new.dropna(inplace=True)

# Split category
cat_split = df_new['category'].str.split('|', expand=True)
cat_split = cat_split.rename(columns={0:'Main category', 1:'Sub category'})
df_new['category'] = cat_split['Main category']

# exclude incorrect character
count = df_new['rating'].str.contains('\|').sum()
df_new = df_new[df_new['rating'].apply(lambda x: '|' not in str(x))]

# changing data types
df_new['rating'] = df_new['rating'].astype(str).str.replace(',', '').astype(float)

# drop data 
dataset = df_new.copy()
dataset = dataset.drop(columns=['product_name', 'category', 'discounted_price', 'actual_price',
                              'discount_percentage', 'about_product', 'user_name', 'review_id',
                              'review_title', 'review_content', 'img_link', 'product_link', 'rating_count'])


# encoding user_id & product_id
# Encode user_id using StringLookup
user_lookup = tf.keras.layers.StringLookup(
    vocabulary=list(dataset['user_id'].unique()),  # List of unique user_ids
    mask_token=None,
    num_oov_indices=0,
    output_mode='int',  # Output integers
    name='user_lookup'
)

# Transform user_id to integers
dataset['encoded_user_id'] = user_lookup(dataset['user_id'])

# Encode product_id using StringLookup
product_lookup = tf.keras.layers.StringLookup(
    vocabulary=list(dataset['product_id'].unique()),  # List of unique user_ids
    mask_token=None,
    num_oov_indices=0,
    output_mode='int',  # Output integers
    name='product_lookup'
)

# Transform product_id to integers
dataset['encoded_product_id'] = product_lookup(dataset['product_id'])

new_data = dataset.copy()
new_data = new_data.drop('user_id', axis=1)

X_product = new_data[['encoded_product_id']]
X_rating = new_data[['rating']]
y = new_data['rating']

X_prod_train, X_prod_test, X_rating_train, X_rating_test, y_train, y_test = train_test_split(
    X_product, X_rating, y, test_size=0.2, random_state=42)

# Normalisasi fitur menggunakan StandardScaler
scaler = StandardScaler()
X_prod_train_scaled = scaler.fit_transform(X_prod_train)
X_prod_test_scaled = scaler.transform(X_prod_test)


class RecommendationRequest(BaseModel):
    encoded_user_id: int

class RecommendationResponse(BaseModel):
    user_name: str
    recommendations: list

def recommend_products(encoded_user_id, dataset, top_k_rating=5):
    # Filter dataset berdasarkan user_id dan encoded_user_id
    user_data = dataset[(dataset['encoded_user_id'] == encoded_user_id)].drop_duplicates(subset='user_id')

    # Inisialisasi DataFrame untuk hasil rekomendasi
    recommended_products = pd.DataFrame(columns=['product_id', 'product_name', 'rating_x', 'img_link'])

    # Ambil daftar kategori unik
    unique_categories = dataset['category'].unique()

    # Shuffle urutan kategori
    shuffled_categories = np.random.permutation(unique_categories)

    # Inisialisasi DataFrame sebelum loop dimulai
    recommended_products = pd.DataFrame(columns=['product_id', 'product_name', 'rating_x', 'img_link'])

    # Loop melalui setiap kategori
    for category in shuffled_categories:
        # Filter dataset berdasarkan kategori
        category_data = dataset[dataset['category'] == category]

        # Ambil sampel acak dari produk dalam kategori
        random_products = category_data.sample(n=min(top_k_rating, len(category_data))).reset_index(drop=True)

        # Gabungkan hasil rekomendasi untuk kategori saat ini ke dalam DataFrame utama
        recommended_products = pd.concat([recommended_products, random_products])

    # Urutkan hasil rekomendasi berdasarkan rating tertinggi
    recommended_products = recommended_products.sort_values(by='rating_x', ascending=False).reset_index(drop=True).head(top_k_rating)

    return user_data[['user_name']], random_products[['product_id', 'product_name', 'rating_x', 'product_link']]

# Endpoint API
@app.post("/recommend")
def recommend(request: RecommendationRequest):
    encoded_user_id = request.encoded_user_id
    top_k_rating = 5

    # Pemrosesan dataset dilakukan di sini (mirroring the preprocessing steps from your initial script)
    dataset_example = pd.merge(df_new, dataset, on=['product_id', 'user_id'], how='inner')
    user_data, recommended_products = recommend_products(encoded_user_id, dataset_example)

    # Convert the result to Pydantic model
    result = {
        'user_data': user_data.to_dict(orient='records'),
        'recommended_products': recommended_products.to_dict(orient='records')
    }

    return result

if __name__ == "__main__":
    uvicorn.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 3030)))