import csv
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt 



dates = list(range(251))
price = []

def get_data(filename):
    with open(filename, "r") as file:
        Reader = csv.reader(file)
        next(Reader)
        for row in Reader:
            price.append( float(row[1] ) )
    return

def predict_price(dates, price, x):
    dates = np.reshape( dates, (251,1) )

    
    lin = SVR( kernel="linear", C=1e3)
    poly = SVR( kernel="poly", C=1e3, degree=2)
    rbf = SVR( kernel="rbf", C=1e3, gamma=0.1)

    lin.fit(dates,price)    
    poly.fit(dates,price)
    #rbf.fit(dates,price)

    plt.scatter(dates, price, color = "black", label = "Data")
    plt.plot(dates, lin.predict(dates), color = "red", label = "Linear")
    plt.plot(dates, poly.predict(dates), color = "green", label = "Polynomial")
    plt.plot(dates, rbf.predict(dates), color = "blue", label = "RBF")
    plt.xlabel("Dates")
    plt.ylabel("Prices")
    plt.legend()
    plt.show()

    return  lin.predict(x)[0], poly.predict(x)[0], rbf.predict(x)[0]


get_data("GOOGL.csv")

print( predict_price(dates,price,251) )

