def zara(message):
    if message == "hi":
        print("Zara: Hiii Saara! 💙")
    elif message == "sad":
        print("Zara: Aww 🥺 I'm here for you!")
    elif message == "happy":
        print("Zara: YAYYY! Keep smiling! 🎉")
    elif message == "bye":
        print("Zara: Byeee! Come back soon 😌")
    else:
        print("Zara: Hmm... I don't understand that yet 🤔")


while True:
    message = input("You: ").lower()

    if message == "bye":
        zara(message)
        break

    zara(message)