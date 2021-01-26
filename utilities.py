import json

def save_json(data, location = "file"):
    print(f"Saving JSON to {location}.json")
    with open(location+".json", "w") as f:
        json.dump(data, f, indent = 4, sort_keys=True)
    print("Save success")

def load_json(location = "file"):
    data = {}
    print(f"Loading JSON from {location}.json")
    with open(location+".json","r") as f:
        data = json.load(f)
    return data
    print("Load success")

def record_text(text, name):
    with open(name+".txt", "a+") as f:
        f.write(text)
        print(f"Recorded text at {name}")

def save_loss_tape(loss_tape, name, type="w"):
    text = ""
    for loss in loss_tape:
        text += str(loss) + "\n"
    with open(name+".txt", type) as f:
            f.write(text)

