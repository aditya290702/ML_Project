import pandas as pd

def EDA(Data):
    Data_Describe = Data.describe()
    print()
    print("EDA :")
    print(Data_Describe)

    Data = Data[Data["Breed"] == "Jersey"]                   # Filtering rows where Breed is "Jersey"
    Data = Data[Data["Previous_Mastits_status"] == 0]        # Filtering rows where Previous_Mastits_status is 0
    Data = Data.drop(["Breed", "House Number", "Address", "Previous_Mastits_status"], axis=1)   # Dropping unnecessary columns
    return Data

def DATA():
    Data = pd.read_csv("clinical_mastitis_cows.csv")
    Data = EDA(Data)
    print(Data)

def main():
    DATA()

if __name__ == '__main__':
    main()
