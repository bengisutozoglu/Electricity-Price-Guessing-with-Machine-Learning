from tkinter import *
from tkinter.ttk import Combobox
import pickle
import pandas as pd


knnModel_f = open("models/knnModel.pickle","rb")
knnModel = pickle.load(knnModel_f)
knnModel_f.close

gradientModel_f = open("models/GradientModel.pickle","rb")
gradientModel = pickle.load(gradientModel_f)
gradientModel_f.close

MLP_f = open("models/MLPModel.pickle","rb")
MLPModel = pickle.load(MLP_f)
MLP_f.close

rfModel_f = open("models/rfModel.pickle","rb")
rfModel = pickle.load(rfModel_f)
rfModel_f.close

SVR_f = open("models/SVRModel.pickle","rb")
SVRModel = pickle.load(SVR_f)
SVR_f.close

xgbModel_f = open("models/XGBModel.pickle","rb")
XGBModel = pickle.load(xgbModel_f)
xgbModel_f.close

lgbModel_f = open("models/XGBModel.pickle","rb")
lgbModel = pickle.load(lgbModel_f)
lgbModel_f.close

def reg():
    gun = int(listekutusu_gun.get())
    mevsim = listekutusu_mevsim.get()
    if mevsim == "1)Yaz":
        mevsim = 1
    elif mevsim=="2)Sonbahar":
        mevsim = 2
    elif mevsim == "3)Kış":
        mevsim = 3
    elif mevsim == "4)Ilkbahar":
        mevsim = 4
    talep = int(talep_entry.get())
    uretim = int(uretim_entry.get())
    dogalgaz = int(dogalgaz_entry.get())
    model = listekutusu_reg.get()
    if model == "knnModel":
        model = knnModel
    elif model == "gradientModel":
        model = gradientModel
    elif model == "MLPModel":
        model = MLPModel
    elif model == "rfModel":
        model = rfModel
    elif model == "SVRModel":
        model = SVRModel
    elif model == "XGBModel":
        model = XGBModel
    elif model == "lgbModel":
        model = lgbModel

    dic = {
        "gün":gun,
        "mevsim":mevsim,
        "talep":talep,
        "gazfiyat":dogalgaz,
        "uretim":uretim }
    df = pd.DataFrame()
    df = df.append(dic,ignore_index=True)
    pred = model.predict(df)
    sonuc["text"] = pred
    
    
        


pencere = Tk()
pencere.title("Elektrik Fiyat Tahmin Tablosu")
pencere.state()
pencere.geometry("500x350")

yazi_gun = Label(pencere, text="Gunu Seciniz:", justify="left", anchor="nw", font=("Times New Roman", "12"))
yazi_gun.pack(anchor="nw")
liste_gun = [1,2,3,4,5,6,7]
listekutusu_gun = Combobox(pencere, values=liste_gun)
listekutusu_gun.set(1)
listekutusu_gun.pack(anchor="nw")

# Mevsim seçme:
yazi_mevsim = Label(pencere, text="Mevsimi Seciniz:", justify="left", anchor="nw", font=("Times New Roman", "12"))
yazi_mevsim.pack(anchor="nw")
liste_mevsim = ["1)Yaz", "2)Sonbahar", "3)Kış", "4)Ilkbahar"]
listekutusu_mevsim = Combobox(pencere, values=liste_mevsim)
listekutusu_mevsim.set("1)Yaz")
listekutusu_mevsim.pack(anchor="nw")

# Talep girme:
yazi_talep = Label(pencere, text="Talep Miktarini Giriniz:", justify="left", anchor="nw",
                   font=("Times New Roman", "12"))
yazi_talep.pack(anchor="nw")
talep_entry = Entry()
talep_entry.pack(anchor="nw")

# Üretim miktarı girme:
yazi_uretim = Label(text="Toplam Uretimi Giriniz:", justify="left", anchor="nw", font=("Times New Roman", "12"))
yazi_uretim.pack(anchor="nw")
uretim_entry = Entry()
uretim_entry.pack(anchor="nw")

# Doğalgaz fiyatı girme:
yazi_dogalgaz = Label(text="Dogalgaz Fiyatini Giriniz:", justify="left", anchor="nw", font=("Times New Roman", "12"))
yazi_dogalgaz.pack(anchor="nw")
dogalgaz_entry = Entry()
dogalgaz_entry.pack(anchor="nw")

#Regressor Belirleme
yazi_reg = Label(pencere, text="Regressor Seciniz:", justify="left", anchor="nw", font=("Times New Roman", "12"))
yazi_reg.pack(anchor="nw")
liste_reg = ["knnModel", "gradientModel", "MLPModel","rfModel","SVRModel","XGBModel","lgbModel"]
listekutusu_reg = Combobox(pencere, values=liste_reg)
listekutusu_reg.set("1)Model")
listekutusu_reg.pack(anchor="nw")

buton = Button(pencere,text="Fiyat",command=reg)
buton.pack()

sonuc = Label(text="Sonuc:")
sonuc.pack()

pencere.mainloop()
