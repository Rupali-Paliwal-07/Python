from sklearn import tree

def BallPredictor():

    Features = [[35,1],[47,1],[90,0],[48,0],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]
    Labels = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

    obj = tree.DecisionTreeClassifier()

    obj = obj.fit(Features,Labels) #training

    print(obj.predict([97,0])
def main():
    print("______________Ball predictor Case study_____________")

    BallPredictor()

if __name__ =="__main__":
    main()
