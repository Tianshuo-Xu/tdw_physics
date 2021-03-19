from tdw.controller import Controller

c = Controller()
print("Everything is OK!")
c.communicate({"$type": "terminate"})