from sklearn import tree

def BallPredictor(weight,surface):

    Features = [[35,1],[47,1],[90,0],[48,0],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]
    Labels = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

    obj = tree.DecisionTreeClassifier()

    obj = obj.fit(Features,Labels) #training

    ret = (obj.predict([[weight,surface]]))
    if ret == 1:
        print("Your object is Tennis ball")
    else:
        print("Your ball is Cricket ball")
def main():
    print("_______________Ball Predictor Case Study_________________")

    print("Please enter the weight of your object in grams")
    weight = int(input())
    print("Please enter the type of surface of your object(Rough/Smooth)")
    surface = input()
    if surface.lower() == "rough":
        surface = 1
    elif surface.lower() == "smooth":
        surface = 0
    else:
        print("Invalid type of surface")
        exit()
    BallPredictor(weight,surface)

if __name__ =="__main__":
    main()
