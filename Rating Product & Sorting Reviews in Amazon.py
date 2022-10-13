# İş Problemi
#E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış
#sonrası verilen puanların doğru şekilde hesaplanmasıdır. Bu
#problemin çözümü e-ticaret sitesi için daha fazla müşteri
#memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
#alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer
#problem ise ürünlere verilen yorumların doğru bir şekilde
#sıralanması olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne
#çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem
#maddi kayıp hem de müşteri kaybına neden olacaktır. Bu 2 temel
#problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını
#arttırırken müşteriler ise satın alma yolculuğunu sorunsuz olarak
#tamamlayacaktır.

#Veri Seti

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


df = pd.read_csv("datasets/amazon_review.csv")
df.head()
df.shape
df.info()

#######################################################################################################################
#Görev 1:  Average Rating’i güncel yorumlara göre hesaplayınız ve var olan average rating ile kıyaslayınız.

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır. \
# Bu görevde amacımız verilen puanları tarihegöre ağırlıklandırarakdeğerlendirmek.\
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.

# Adım 1:   Ürünün ortalama puanını hesaplayınız.

df["overall"].mean()
# 4.587589013224822

# Adım 2: Tarihe göre ağırlıklı puan ortalamasını hesaplayınız.

#review Time değişkenini tarih değişkeni olarak tanıtmanız \

df["reviewTime"] = pd.to_datetime(df["reviewTime"])
df.info()

#reviewTime'ın max değerini current_date olarak kabul etmeniz\

df["reviewTime"].max()
current_date = pd.to_datetime(df["reviewTime"].max())

#her bir puan-yorum tarihi ile current_date'in farkını gün cinsinden ifade ederek yeni değişken oluşturmanız \
# ve gün cinsinden ifade edilen değişkeni quantile fonksiyonu ile 4'e bölüp (3 çeyrek verilirse 4 parça çıkar)\
# çeyrekliklerden gelen değerlere göre ağırlıklandırma yapmanız gerekir. \
# Örneğin q1 = 12 ise ağırlıklandırırken 12 günden az süre önce yapılan yorumların ortalamasını alıp bunlara yüksek ağırlık vermek gibi.

df["days"] = (current_date - df["reviewTime"]).dt.days
df.head()

q1 = df["days"].quantile(0.25)
q1 #280
q2 = df["days"].quantile(0.50)
q2 #430
q3 = df["days"].quantile(0.75)
q3 #600


def time_based_weighted_average(df, w1=28, w2=26, w3=24, w4=22):
    return df.loc[df["days"] <= 280, "overall"].mean() * w1/100 + \
           df.loc[(df["days"] > 280) & (df["days"] <= 430), "overall"].mean() * w2/100 + \
           df.loc[(df["days"] > 430) & (df["days"] <= 600), "overall"].mean() * w3/100 + \
           df.loc[df["days"] > 600, "overall"].mean() * w4/100

time_based_weighted_average(df)
# 4.595593165128118

# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.

df.loc[df["days"] <= 280, "overall"].mean()
# 4.6957928802588995

df.loc[(df["days"] > 280) & (df["days"] <= 430), "overall"].mean()
# 4.636140637775961

df.loc[(df["days"] > 430) & (df["days"] <= 600), "overall"].mean()
# 4.571661237785016

df.loc[df["days"] > 600, "overall"].mean()
# 4.4462540716612375

# Beklenildiği gibi güncel tarihe yakın yapılan yorumların ortalamaya katkısı daha fazladır.



#######################################################################################################################
#Görev 2 : Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.

# Adım 1:  helpful_no değişkenini üretiniz.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head()

# total_vote bir yoruma verilen toplam up-down sayısıdır.\
# up, helpful demektir.\
# Veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.\
# Toplam oy sayısından (total_vote) yararlı oy sayısı (helpful_yes) çıkarılarak yararlı bulunmayan oy sayılarını (helpful_no) bulunuz.

# Adım 2:  score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyiniz.

# score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayabilmek için score_pos_neg_diff,\
# score_average_rating ve wilson_lower_bound fonksiyonlarını tanımlayınız.
# score_pos_neg_diff'a göre skorlar oluşturunuz. Ardından; df içerisinde score_pos_neg_diff ismiyle kaydediniz.\
# score_average_rating'a göre skorlar oluşturunuz. Ardından; df içerisinde score_average_rating ismiyle kaydediniz.
# wilson_lower_bound'a göre skorlar oluşturunuz. Ardından; df içerisinde wilson_lower_bound ismiyle kaydediniz.


def score_pos_neg_diff(helpful_yes, helpful_no):
    return helpful_yes - helpful_no


def score_average_rating(helpful_yes, helpful_no):
    if helpful_yes + helpful_no == 0:
        return 0
    return helpful_yes / (helpful_yes + helpful_no)


def wilson_lower_bound(helpful_yes, helpful_no, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = helpful_yes + helpful_no
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * helpful_yes / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"],
                                                                   x["helpful_no"]), axis=1)

# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],
                                                                     x["helpful_no"]), axis=1)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],
                                                                 x["helpful_no"]), axis=1)


df.head()

# Adım 3:  20 Yorumu belirleyiniz ve sonuçları Yorumlayınız.

# wilson_lower_bound'a göre ilk 20 yorumu belirleyip sıralayanız.

df.sort_values("wilson_lower_bound", ascending= False).head(20)

# Sonuçları yorumlayınız.

#helpful yes ile helpful no arasındaki fark ne kadar büyükse ve total vote sayısı ne kadar fazlaysa wilson lower bound değişkeni de o kadar yüksektir.


















