def zara(name, mood):
    print("Hello", name, "! 🤖")

    if mood == "happy":
        print("Yayyy! Keep smiling! 💙")
    elif mood == "sad":
        print("Aww 🥺 Don't worry, I'm here for you!")
    elif mood == "angry":
        print("Uh oh 😳 Who made my boss angry?")
    elif mood == "lazy":
        print("GET UP! 😈 Your dreams aren't going to code themselves!")
    else:
        print("Hmm... interesting mood 😎")


name = input("Zara: What's your name? ")
mood = input("Zara: How are you feeling? ")

zara(name, mood)