import pandas as pd
import statsmodels.api as sm

veri = pd.read_csv('Data/otomobil.csv')

# Bağımlı değişkeni bağımsız değişkenlerden çıkarıyoruz.
bagimsiz_degiskenler = veri.drop(['SatisFiyati'], axis=1)

# Bağımlı değişkeni seçiyoruz
bagimli_degisken = veri['SatisFiyati']

# Sabit (intercept) terimini ekliyoruz
bagimsiz_degiskenler = sm.add_constant(bagimsiz_degiskenler)

# Lineer regresyon modelini oluşturuyoruz.
model = sm.OLS(bagimli_degisken, bagimsiz_degiskenler)

sonuclar = model.fit()

# Model istatistiklerini görünrülüyoruz.
print(sonuclar.summary())