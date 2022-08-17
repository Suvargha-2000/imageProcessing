from PIL import Image
import os 
files = os.listdir("./images/train")

lowestHeight = 42
lowestWidth = 32

counter = 0

for i in files :
    counter += 1
    if counter % 10 == 0 : 
        continue
    img = Image.open('images/train/'+i)
    img = img.resize((32, 32))
    img.save('./compImages/'+i)

# print("height : " + str(lowestHeight))
# print("width : " + str(lowestWidth))